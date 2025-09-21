# src/data/collect/fetch_pjm_load.py
# usage: PJM_CLIENT_ID=... PJM_CLIENT_SECRET=... python -m src.data.collect.fetch_pjm_load --start 2018-01-01 --end 2025-09-01 --out data/raw/load.csv
import os, argparse, requests, pandas as pd
from datetime import datetime

TOKEN_URL = "https://api.pjm.com/oauth2/access_token"
DATA_URL  = "https://api.pjm.com/api/v1/operationaldemand"  # example dataset; PJM has many (check docs)

def get_token(cid, secret):
    r = requests.post(TOKEN_URL, data={"grant_type":"client_credentials"}, auth=(cid, secret), timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]

def fetch_pjm(start: str, end: str, token: str) -> pd.DataFrame:
    params = {"download":"true", "startRow":1, "rowCount":10000, "startDate":start, "endDate":end}
    headers = {"Authorization": f"Bearer {token}", "Accept":"application/json"}
    rows = []
    offset = 1
    while True:
        params["startRow"] = offset
        r = requests.get(DATA_URL, params=params, headers=headers, timeout=60)
        r.raise_for_status()
        js = r.json()
        data = js.get("items", [])
        if not data: break
        rows.extend(data)
        if len(data) < params["rowCount"]: break
        offset += params["rowCount"]
    df = pd.DataFrame(rows)
    # typical columns include 'occurred' or 'datetime_beginning_utc' etc. harmonize to timestamp/load_mw
    ts_col = next(c for c in df.columns if "time" in c.lower() or "occur" in c.lower())
    val_col = next(c for c in df.columns if "mw" in c.lower() or "load" in c.lower() or "value" in c.lower())
    df["timestamp"] = pd.to_datetime(df[ts_col], utc=True)
    df["load_mw"] = pd.to_numeric(df[val_col], errors="coerce")
    return df[["timestamp","load_mw"]].dropna().sort_values("timestamp")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default=datetime.utcnow().strftime("%Y-%m-%d"))
    ap.add_argument("--out", default="data/raw/load.csv")
    args = ap.parse_args()
    cid = os.getenv("PJM_CLIENT_ID"); secret = os.getenv("PJM_CLIENT_SECRET")
    if not (cid and secret): raise SystemExit("set PJM_CLIENT_ID and PJM_CLIENT_SECRET (free PJM Data Miner account)")
    tok = get_token(cid, secret)
    df = fetch_pjm(args.start, args.end, tok)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out} with {len(df)} rows")
