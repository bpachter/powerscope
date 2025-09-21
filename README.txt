# PowerScope - Electricity Demand Forecasting

PowerScope forecasts electricity demand with credible P10/P50/P90 bands using public ISO + weather data. Cut surprises, purchase smarter, and schedule work when risk is low.

## Features

- **Quantile Forecasting**: Generate P10, P50, and P90 demand predictions
- **Multiple Models**: LightGBM and PyTorch neural networks
- **Weather Integration**: Incorporate weather data from Meteostat
- **Conformal Prediction**: Calibrated uncertainty quantification
- **REST API**: FastAPI service for real-time forecasting
- **Streamlit App**: Interactive web interface
- **Comprehensive Evaluation**: Time-based metrics and visualizations

## Project Structure

```
powerscope/
├── app/
│   └── streamlit_app.py          # Streamlit web application
├── conf/
│   └── config.yaml               # Configuration settings
├── data/
│   ├── raw/                      # Raw data files
│   └── processed/                # Processed data files
├── src/
│   ├── calibration/
│   │   └── conformal.py          # Conformal prediction methods
│   ├── data/
│   │   ├── iso_local_csv.py      # ISO data loading
│   │   └── weather_meteostat.py  # Weather data from Meteostat
│   ├── features/
│   │   └── build_features.py     # Feature engineering
│   ├── models/
│   │   ├── lightgbm_quantile.py  # LightGBM quantile regression
│   │   └── torch_models.py       # PyTorch neural networks
│   ├── utils/
│   │   ├── io.py                 # Input/output utilities
│   │   ├── metrics.py            # Evaluation metrics
│   │   └── roll.py               # Rolling window utilities
│   ├── serve/
│   │   └── api.py                # FastAPI service
│   ├── train.py                  # Model training script
│   └── evaluate.py               # Model evaluation script
├── artifacts/
│   ├── models/                   # Trained model files
│   └── reports/                  # Evaluation reports and plots
├── requirements.txt              # Python dependencies
└── README.txt                    # This file
```

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure the system**:
   Edit `conf/config.yaml` to set your data sources and model parameters.

3. **Train models**:
   ```bash
   python src/train.py --config conf/config.yaml --model both
   ```

4. **Evaluate models**:
   ```bash
   python src/evaluate.py --config conf/config.yaml --model both --plots
   ```

5. **Start the API service**:
   ```bash
   python src/serve/api.py
   ```
   The API will be available at http://localhost:8000 with documentation at http://localhost:8000/docs

6. **Run the Streamlit app**:
   ```bash
   streamlit run app/streamlit_app.py
   ```

## Configuration

The `conf/config.yaml` file contains all configuration settings:

- **Data sources**: ISO CSV files and weather API settings
- **Model parameters**: LightGBM and PyTorch hyperparameters
- **Training settings**: Test/validation splits, random seeds
- **Forecasting**: Quantile levels and forecast horizons

## Models

### LightGBM Quantile Regression
- Fast training and inference
- Built-in feature importance
- Robust to missing data
- Excellent for tabular data

### PyTorch Neural Networks
- Deep quantile regression
- LSTM support for sequential data
- Flexible architecture
- Custom loss functions

## Data Sources

### ISO Load Data
Place CSV files with electricity demand data in `data/raw/iso_data/`. Expected columns:
- `timestamp`: DateTime index
- `demand`: Electricity demand in MW

### Weather Data
PowerScope can automatically fetch weather data from Meteostat API. Configure station locations in `conf/config.yaml`.

## API Usage

### Forecast Endpoint
```bash
curl -X POST "http://localhost:8000/forecast" \
     -H "Content-Type: application/json" \
     -d '{
       "start_datetime": "2024-01-01T00:00:00",
       "end_datetime": "2024-01-02T00:00:00",
       "model_type": "lightgbm",
       "confidence_intervals": true
     }'
```

### Health Check
```bash
curl "http://localhost:8000/health"
```

## Evaluation Metrics

PowerScope provides comprehensive evaluation including:

- **Point Forecasts**: MAE, RMSE, MAPE, SMAPE, R²
- **Quantile Forecasts**: Quantile loss, coverage probability
- **Interval Metrics**: Interval width, interval score
- **Time-based Analysis**: Performance by hour and day of week
- **Conformal Prediction**: Calibrated prediction intervals

## Features

### Time Features
- Hour, day, month, year
- Day of week, day of year
- Cyclical encoding (sin/cos)
- Weekend indicators
- Peak hour flags

### Lag Features
- Recent lags (1-3 hours)
- Daily lags (24, 48 hours)
- Weekly lags (168 hours)

### Rolling Features
- Moving averages
- Rolling standard deviation
- Rolling min/max
- Z-scores and percentile ranks

### Weather Features
- Temperature features
- Cooling/heating degree days
- Weather interactions with time

## Uncertainty Quantification

PowerScope provides multiple approaches to uncertainty:

1. **Quantile Regression**: Direct prediction of multiple quantiles
2. **Conformal Prediction**: Calibrated prediction intervals
3. **Model Ensembles**: Combining multiple model predictions

## Development

### Training New Models
```bash
python src/train.py --config conf/config.yaml --model lightgbm
python src/train.py --config conf/config.yaml --model torch
```

### Running Evaluations
```bash
python src/evaluate.py --config conf/config.yaml --plots --conformal
```

### Testing the API
```bash
# Start the API
python src/serve/api.py

# Test in another terminal
curl "http://localhost:8000/forecast/sample"
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For questions or issues, please:
1. Check the documentation
2. Review existing issues
3. Create a new issue with detailed information

## Roadmap

- [ ] Additional model types (Prophet, ARIMA)
- [ ] Real-time data streaming
- [ ] Model deployment to cloud platforms
- [ ] Advanced ensemble methods
- [ ] Multi-region forecasting
- [ ] Anomaly detection integration