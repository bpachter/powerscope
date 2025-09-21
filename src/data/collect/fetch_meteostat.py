# src/data/collect/fetch_meteostat.py
# usage: python -m src.data.collect.fetch_meteostat --lat 40.49 --lon -80.23 --start 2018-01-01 --end 2025-09-01 --out data/processed/weather.parquet --tz America/New_York
import argparse, pandas as pd
from meteostat import Hourly, Point

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", default="data/processed/weather.parquet")
    ap.add_argument("--tz", default="America/New_York")
    args = ap.parse_args()
    pt = Point(args.lat, args.lon)
    start = pd.to_datetime(args.start)
    end = pd.to_datetime(args.end) if args.end else pd.Timestamp.utcnow()
    df = Hourly(pt, start, end).fetch()
    df = df.tz_localize("UTC").tz_convert(args.tz)
    df["timestamp"] = df.index
    df = df.reset_index(drop=True)
    df.to_parquet(args.out, index=False)
    print(f"wrote {args.out} with {len(df)} rows")
