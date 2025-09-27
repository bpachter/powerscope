# src/data/collect/fetch_eia_load.py
# usage: EIA_API_KEY=... python -m src.data.collect.fetch_eia_load --ba PJM --start 2018-01-01 --end 2025-09-01 --out data/raw/load.csv
import os, argparse, pandas as pd, requests
from datetime import datetime
from dotenv import load_dotenv  # Add this import

# Load environment variables from .env file
load_dotenv()

DATASET = "electricity/rto/region-data"  # eia v2 dataset for hourly demand

def fetch_eia_hourly(ba: str, start: str, end: str, api_key: str) -> pd.DataFrame:
    # note: endpoint/dim names evolve. if 4xx, check EIA v2 docs for the exact dataset + facet names.
    base = "https://api.eia.gov/v2"
    url = f"{base}/{DATASET}/data/"
    params = {
        "api_key": api_key,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": ba,   # e.g., "PJM", "NYIS", "MISO", ERCOT (ERCO), CAISO (CISO)
        "facets[type][]": "D",        # D = Demand
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "start": f"{start}T00:00",
        "end": f"{end}T00:00",
        "offset": 0,
        "length": 5000
    }
    rows = []
    while True:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        js = r.json()
        data = js.get("data", [])
        if not data: break
        rows.extend(data)
        params["offset"] += params["length"]
    df = pd.DataFrame(rows)
    # normalize columns the pipeline expects
    df["timestamp"] = pd.to_datetime(df["period"], utc=True)
    df["load_mw"] = pd.to_numeric(df["value"], errors="coerce")
    return df[["timestamp","load_mw"]].dropna().sort_values("timestamp")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ba", default="PJM")                 # BA codes: PJM, NYIS, MISO, ERCOT (ERCO), CAISO (CISO)
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default=datetime.utcnow().strftime("%Y-%m-%d"))
    ap.add_argument("--out", default="data/raw/load.csv")
    args = ap.parse_args()
    
    # Now it will automatically load from .env file
    key = os.getenv("EIA_API_KEY")
    if not key: 
        raise SystemExit("EIA_API_KEY not found. Create .env file with: EIA_API_KEY=your_key_here")
    
    df = fetch_eia_hourly(args.ba, args.start, args.end, key)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out} with {len(df)} rows")
