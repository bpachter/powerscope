import pandas as pd

def load_local_csv(path: str, tz: str):
    # expects columns: timestamp, load_mw
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce').dt.tz_convert(tz)
    df = df.sort_values('timestamp').dropna()
    # resample to hourly if needed
    df = df.set_index('timestamp').resample('H').mean().ffill().reset_index()
    df.rename(columns={'index':'timestamp'}, inplace=True)
    return df[['timestamp','load_mw']]
