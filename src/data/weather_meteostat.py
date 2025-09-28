import pandas as pd
from meteostat import Hourly, Point
from datetime import datetime

def fetch_weather(stations, start: str, end: str | None, tz: str):
    # stations: list of dicts with lat, lon
    start_dt = pd.to_datetime(start).tz_localize(None)  # Make timezone-naive
    end_dt = pd.to_datetime(end).tz_localize(None) if end else pd.Timestamp.utcnow().tz_localize(None)
    frames = []
    for s in stations:
        pt = Point(s['lat'], s['lon'])
        data = Hourly(pt, start_dt, end_dt).fetch()
        data = data.tz_localize('UTC').tz_convert(tz)
        data = data.rename(columns={
            'temp': f"temp_{s['name']}",
            'dwpt': f"dwpt_{s['name']}",
            'rhum': f"rhum_{s['name']}",
            'wspd': f"wspd_{s['name']}",
            'prcp': f"prcp_{s['name']}"
        })
        data['timestamp'] = data.index
        frames.append(data.reset_index(drop=True))
    # join on timestamp (outer then mean over stations)
    merged = None
    for i, f in enumerate(frames):
        cols = ['timestamp'] + [c for c in f.columns if c != 'timestamp']
        f = f[cols]
        merged = f if merged is None else pd.merge(merged, f, on='timestamp', how='outer')
    # aggregate station features by mean
    agg = {}
    for base in ['temp','dwpt','rhum','wspd','prcp']:
        cols = [c for c in merged.columns if c.startswith(base+'_')]
        if cols:
            agg[base] = merged[cols].mean(axis=1)
    out = pd.DataFrame({'timestamp': merged['timestamp']}).assign(**agg)
    out = out.sort_values('timestamp').dropna(subset=['temp']).reset_index(drop=True)
    return out
