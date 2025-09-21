"""
FastAPI service for PowerScope electricity demand forecasting.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

from src.utils.io import load_config, load_pickle
from src.data.iso_local_csv import ISOLocalCSVLoader
from src.data.weather_meteostat import WeatherMeteostatLoader
from src.features.build_features import FeatureBuilder
from src.models.lightgbm_quantile import LightGBMQuantileRegressor
from src.models.torch_models import TorchQuantileTrainer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PowerScope API",
    description="Electricity demand forecasting with credible P10/P50/P90 bands",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and config
config = None
lightgbm_model = None
torch_model = None
feature_builder = None


# Pydantic models for API
class ForecastRequest(BaseModel):
    """Request model for forecasting."""
    start_datetime: str
    end_datetime: str
    model_type: str = "lightgbm"
    include_weather: bool = True
    confidence_intervals: bool = True
    
    @validator('model_type')
    def validate_model_type(cls, v):
        if v not in ['lightgbm', 'torch']:
            raise ValueError('model_type must be "lightgbm" or "torch"')
        return v
    
    @validator('start_datetime', 'end_datetime')
    def validate_datetime(cls, v):
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
        except ValueError:
            raise ValueError('Invalid datetime format. Use ISO format (YYYY-MM-DDTHH:MM:SS)')
        return v


class ForecastPoint(BaseModel):
    """Single forecast point."""
    timestamp: str
    demand_mw: float
    demand_p10: Optional[float] = None
    demand_p50: Optional[float] = None
    demand_p90: Optional[float] = None


class ForecastResponse(BaseModel):
    """Response model for forecasting."""
    status: str
    model_type: str
    forecast_horizon_hours: int
    timestamps: List[str]
    forecasts: List[ForecastPoint]
    metadata: Dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    models_loaded: Dict[str, bool]
    version: str


class ModelStatus(BaseModel):
    """Model status information."""
    model_type: str
    loaded: bool
    last_updated: Optional[str] = None
    performance_metrics: Optional[Dict[str, float]] = None


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Load models and configuration on startup."""
    global config, lightgbm_model, torch_model, feature_builder
    
    logger.info("Starting PowerScope API...")
    
    # Load configuration
    try:
        config = load_config("conf/config.yaml")
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        config = {}
    
    # Initialize feature builder
    feature_builder = FeatureBuilder()
    
    # Load models
    await load_models()
    
    logger.info("PowerScope API startup complete")


async def load_models():
    """Load trained models."""
    global lightgbm_model, torch_model
    
    # Load LightGBM model
    try:
        lightgbm_path = "artifacts/models/lightgbm_model.pkl"
        if Path(lightgbm_path).exists():
            lightgbm_model = load_pickle(lightgbm_path)
            logger.info("LightGBM model loaded successfully")
        else:
            logger.warning("LightGBM model not found")
    except Exception as e:
        logger.error(f"Failed to load LightGBM model: {e}")
    
    # Load PyTorch model
    try:
        torch_path = "artifacts/models/torch_trainer.pkl"
        if Path(torch_path).exists():
            torch_model = load_pickle(torch_path)
            logger.info("PyTorch model loaded successfully")
        else:
            logger.warning("PyTorch model not found")
    except Exception as e:
        logger.error(f"Failed to load PyTorch model: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("PowerScope API shutting down...")


# Helper functions
def get_historical_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Get historical data for feature engineering.
    
    Args:
        start_date: Start date string
        end_date: End date string
        
    Returns:
        Historical DataFrame
    """
    try:
        # Load ISO data
        iso_loader = ISOLocalCSVLoader(config.get('data', {}).get('iso', {}).get('source_path', 'data/raw/'))
        demand_df = iso_loader.load_demand_data(start_date, end_date)
        
        if demand_df.empty:
            # Create synthetic data for demonstration
            start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            dates = pd.date_range(start_dt, end_dt, freq='h')
            
            # Generate realistic demand pattern
            hour_of_day = dates.hour
            day_of_week = dates.dayofweek
            
            base_demand = 1000
            hourly_pattern = 200 * np.sin(2 * np.pi * hour_of_day / 24 - np.pi/4)  # Peak in evening
            weekly_pattern = 100 * (1 - 0.3 * (day_of_week >= 5))  # Lower on weekends
            noise = np.random.normal(0, 50, len(dates))
            
            demand = base_demand + hourly_pattern + weekly_pattern + noise
            
            demand_df = pd.DataFrame({
                'timestamp': dates,
                'demand': demand
            })
        
        return demand_df
    
    except Exception as e:
        logger.error(f"Error loading historical data: {e}")
        return pd.DataFrame()


def prepare_forecast_features(df: pd.DataFrame, horizon_hours: int) -> pd.DataFrame:
    """
    Prepare features for forecasting.
    
    Args:
        df: Historical data DataFrame
        horizon_hours: Forecast horizon in hours
        
    Returns:
        Features DataFrame
    """
    # Build features
    df_features = feature_builder.build_features(df, 'demand', 'timestamp')
    
    # Create future timestamps
    last_timestamp = df_features['timestamp'].max()
    future_timestamps = pd.date_range(
        start=last_timestamp + timedelta(hours=1),
        periods=horizon_hours,
        freq='h'
    )
    
    # Create future DataFrame with time features only
    future_df = pd.DataFrame({'timestamp': future_timestamps})
    future_df = feature_builder.add_time_features(future_df, 'timestamp')
    
    # For lag and rolling features, use last known values (forward fill)
    feature_cols = [col for col in df_features.columns if col not in ['timestamp', 'demand']]
    
    # Combine historical and future data
    combined_df = pd.concat([df_features, future_df], ignore_index=True)
    
    # Forward fill features for future periods
    combined_df[feature_cols] = combined_df[feature_cols].fillna(method='ffill')
    
    # Return only future features
    future_features = combined_df.tail(horizon_hours)[feature_cols]
    future_features.index = range(len(future_features))
    
    return future_features, future_timestamps


def make_predictions(model, model_type: str, features: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Make predictions using the specified model.
    
    Args:
        model: Trained model
        model_type: Type of model ('lightgbm' or 'torch')
        features: Features DataFrame
        
    Returns:
        Dictionary of predictions
    """
    if model_type == 'lightgbm':
        predictions = model.predict(features)
        return predictions
    
    elif model_type == 'torch':
        features_np = features.values.astype(np.float32)
        pred_array = model.predict(features_np)
        
        quantiles = config.get('forecast', {}).get('quantiles', [0.1, 0.5, 0.9])
        predictions = {q: pred_array[:, i] for i, q in enumerate(quantiles)}
        return predictions
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# API endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "PowerScope Electricity Demand Forecasting API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        models_loaded={
            "lightgbm": lightgbm_model is not None,
            "torch": torch_model is not None
        },
        version="1.0.0"
    )


@app.get("/models/status", response_model=List[ModelStatus])
async def get_model_status():
    """Get status of all loaded models."""
    models = []
    
    # LightGBM status
    models.append(ModelStatus(
        model_type="lightgbm",
        loaded=lightgbm_model is not None,
        last_updated=None,  # Could track this in production
        performance_metrics=None  # Could load from saved evaluation
    ))
    
    # PyTorch status
    models.append(ModelStatus(
        model_type="torch",
        loaded=torch_model is not None,
        last_updated=None,
        performance_metrics=None
    ))
    
    return models


@app.post("/models/reload")
async def reload_models(background_tasks: BackgroundTasks):
    """Reload models in the background."""
    background_tasks.add_task(load_models)
    return {"message": "Model reload initiated"}


@app.post("/forecast", response_model=ForecastResponse)
async def create_forecast(request: ForecastRequest):
    """
    Create electricity demand forecast.
    
    Args:
        request: Forecast request parameters
        
    Returns:
        Forecast response with predictions
    """
    try:
        # Validate model availability
        if request.model_type == 'lightgbm' and lightgbm_model is None:
            raise HTTPException(status_code=503, detail="LightGBM model not available")
        elif request.model_type == 'torch' and torch_model is None:
            raise HTTPException(status_code=503, detail="PyTorch model not available")
        
        # Parse timestamps
        start_dt = datetime.fromisoformat(request.start_datetime.replace('Z', '+00:00'))
        end_dt = datetime.fromisoformat(request.end_datetime.replace('Z', '+00:00'))
        
        # Calculate forecast horizon
        horizon_hours = int((end_dt - start_dt).total_seconds() / 3600)
        
        if horizon_hours <= 0:
            raise HTTPException(status_code=400, detail="End datetime must be after start datetime")
        if horizon_hours > 168:  # Limit to 1 week
            raise HTTPException(status_code=400, detail="Forecast horizon cannot exceed 168 hours")
        
        # Get historical data (last 30 days for context)
        hist_start = start_dt - timedelta(days=30)
        historical_df = get_historical_data(hist_start.isoformat(), start_dt.isoformat())
        
        if historical_df.empty:
            raise HTTPException(status_code=500, detail="Could not load historical data")
        
        # Prepare features
        features, future_timestamps = prepare_forecast_features(historical_df, horizon_hours)
        
        # Select model
        model = lightgbm_model if request.model_type == 'lightgbm' else torch_model
        
        # Make predictions
        predictions = make_predictions(model, request.model_type, features)
        
        # Format response
        forecasts = []
        timestamps = []
        
        for i, timestamp in enumerate(future_timestamps):
            timestamps.append(timestamp.isoformat())
            
            # Get median prediction
            median_pred = predictions.get(0.5)
            if median_pred is None:
                median_pred = list(predictions.values())[len(predictions) // 2]
            
            forecast_point = ForecastPoint(
                timestamp=timestamp.isoformat(),
                demand_mw=float(median_pred[i])
            )
            
            # Add confidence intervals if requested
            if request.confidence_intervals and len(predictions) >= 3:
                sorted_quantiles = sorted(predictions.keys())
                forecast_point.demand_p10 = float(predictions[sorted_quantiles[0]][i])
                forecast_point.demand_p50 = float(predictions.get(0.5, median_pred)[i])
                forecast_point.demand_p90 = float(predictions[sorted_quantiles[-1]][i])
            
            forecasts.append(forecast_point)
        
        # Create metadata
        metadata = {
            "model_version": "1.0.0",
            "features_used": len(features.columns),
            "historical_data_points": len(historical_df),
            "quantiles": list(predictions.keys()) if request.confidence_intervals else [],
            "created_at": datetime.now().isoformat()
        }
        
        return ForecastResponse(
            status="success",
            model_type=request.model_type,
            forecast_horizon_hours=horizon_hours,
            timestamps=timestamps,
            forecasts=forecasts,
            metadata=metadata
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Forecast creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Forecast creation failed: {str(e)}")


@app.get("/forecast/sample")
async def get_sample_forecast():
    """Get a sample forecast for demonstration."""
    # Create sample request
    start_time = datetime.now() + timedelta(hours=1)
    end_time = start_time + timedelta(hours=24)
    
    sample_request = ForecastRequest(
        start_datetime=start_time.isoformat(),
        end_datetime=end_time.isoformat(),
        model_type="lightgbm" if lightgbm_model else "torch",
        include_weather=True,
        confidence_intervals=True
    )
    
    return await create_forecast(sample_request)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)