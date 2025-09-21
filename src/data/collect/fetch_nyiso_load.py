# src/data/collect/fetch_nyiso_load.py
# usage: python -m src.data.collect.fetch_nyiso_load --start 2018-01-01 --end 2025-09-01 --out data/raw/load.csv
# note: NYISO publishes "Actual Load" CSVs; the base URL/file layout can change. This script expects a single CSV URL or a local folder of monthly CSVs.
import argparse, pandas as pd

def merge_csvs(urls_or_paths):
    frames = [pd.read_csv(u) for u in urls_or_paths]
    df = pd.concat(frames, ignore_index=True)
    # harmonize columns (commonly: 'Time Stamp' and 'NYCA' or zonal columns)
    ts_col = next(c for c in df.columns if "time" in c.lower())
    load_col = "NYCA" if "NYCA" in df.columns else next(c for c in df.columns if c.lower() not in [ts_col.lower()])
    df["timestamp"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df["load_mw"] = pd.to_numeric(df[load_col], errors="coerce")
    return df[["timestamp","load_mw"]].dropna().sort_values("timestamp")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="one or more NYISO load CSV URLs/paths")
    ap.add_argument("--out", default="data/raw/load.csv")
    args = ap.parse_args()
    df = merge_csvs(args.inputs)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out} with {len(df)} rows")
