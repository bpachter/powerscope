import numpy as np
import pandas as pd
import holidays as hol

def add_time_features(df: pd.DataFrame, tz: str, fourier_k: int, freq: str = "H"):
    # basic time features
    ts = df['timestamp']
    out = pd.DataFrame({'timestamp': ts})
    out['hour'] = ts.dt.hour
    out['dow'] = ts.dt.dayofweek
    out['month'] = ts.dt.month
    out['weekofyear'] = ts.dt.isocalendar().week.astype(int)
    # fourier seasonal terms (daily and annualish)
    def fourier(t, period, K):
        x = 2*np.pi*np.arange(len(t))/period
        mats = []
        for k in range(1, K+1):
            mats += [np.sin(k*x), np.cos(k*x)]
        return np.vstack(mats).T
    daily = fourier(ts.values, period=24, K=fourier_k)
    yearly = fourier(ts.values, period=24*365.25, K=fourier_k)
    for i in range(daily.shape[1]):
        out[f'sin_cos_daily_{i}'] = daily[:, i]
    for i in range(yearly.shape[1]):
        out[f'sin_cos_yearly_{i}'] = yearly[:, i]
    return out

def add_holiday_features(df: pd.DataFrame, country: str, tz: str):
    # holiday / weekend flags
    cal = hol.country_holidays(country=country)
    out = pd.DataFrame({'timestamp': df['timestamp']})
    out['is_weekend'] = out['timestamp'].dt.dayofweek.isin([5,6]).astype(int)
    out['is_holiday'] = out['timestamp'].dt.date.map(lambda d: 1 if d in cal else 0)
    return out

def build_derived_weather(weather: pd.DataFrame, temp_lags, rolling_hours, use_degree_hours=True):
    df = weather.copy()
    # lags
    for L in temp_lags:
        df[f'temp_lag_{L}h'] = df['temp'].shift(L)
    # rolling means
    for W in rolling_hours:
        df[f'temp_rollmean_{W}h'] = df['temp'].rolling(W, min_periods=1).mean()
    # degree hours (base 18c ~ 65F)
    if use_degree_hours:
        base = 18.0
        df['cooling_deg'] = (df['temp'] - base).clip(lower=0)
        df['heating_deg'] = (base - df['temp']).clip(lower=0)
        for W in [24, 168]:
            df[f'cooling_deg_sum_{W}h'] = df['cooling_deg'].rolling(W, min_periods=1).sum()
            df[f'heating_deg_sum_{W}h'] = df['heating_deg'].rolling(W, min_periods=1).sum()
    return df

def assemble_feature_frame(load_df, weather_df, time_df, holiday_df):
    # align and join all on timestamp
    df = load_df.merge(weather_df, on='timestamp', how='left') \
                .merge(time_df, on='timestamp', how='left') \
                .merge(holiday_df, on='timestamp', how='left')
    df = df.dropna(subset=['load_mw']).sort_values('timestamp').reset_index(drop=True)
    # forward fill weather
    weather_cols = [c for c in df.columns if c not in ['timestamp','load_mw']]
    df[weather_cols] = df[weather_cols].ffill()
    return df
