# probabilistic-load-forecaster (P10/P50/P90)

## what this does
- ingests iso load + weather, builds time/seasonal/weather features
- trains quantile models (lightgbm and pytorch lstm)
- calibrates uncertainty via conformal prediction
- runs rolling-origin backtests with pinball loss, crps, coverage
- serves predictions via fastapi and a streamlit fan chart

## quickstart
1) create env and install
    `pip install -r requirements.txt`
2) get data
- easiest: put a csv at `data/raw/load.csv` with columns: `timestamp,load_mw`
  * timestamp: UTC or local with tz offset (isoformat), hourly cadence
- weather: meteostat is pulled automatically; set your lat/lon(s) in `conf/config.yaml`
- optional: set `PJM_API_KEY` for pjM Data Miner 2; or use local csv first
3) train (lightgbm quantiles)
    `python -m src.train model=lightgbm`
   train (lstm quantiles, uses cuda if available)
   `python -m src.train model=lstm`
4) evaluate (rolling backtest + calibration report)
    `python -m src.evaluate`
5) Serve api
    `uvicorn src.serve.api:app --reload`
   streamlit fan chart
    `streamlit run app/streamlit_app.py`


## data details
- load: provide `data/raw/load.csv` or enable a loader in `conf/config.yaml`
- weather: meteostat pulls hourly features for given stations; cached to `data/processed/weather.parquet`

## outputs
- models and adapters: `artifacts/models/<run_id>/...`
- metrics & plots: `artifacts/reports/<run_id>/...`
- calibrated quantiles (p10/p50/p90) stored alongside model

## notes
- comments in code are terse and start in lower case (by request)
- everything avoids paid services; pjM/nyiso connectors are optional


