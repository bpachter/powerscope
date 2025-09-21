# probabilistic-load-forecaster (P10/P50/P90)

## what this does
- ingests iso load + weather, builds time/seasonal/weather features
- trains quantile models (lightgbm and pytorch lstm)
- calibrates uncertainty via conformal prediction
- runs rolling-origin backtests with pinball loss, crps, coverage
- serves predictions via fastapi and a streamlit fan chart

## Data Sources & Institutions (Context)

PowerScope relies only on **public, $0** sources. Below is what each organization is, why it exists, and exactly what data you’ll pull.

### EIA — U.S. Energy Information Administration
**What it is:** Nonpartisan statistical agency within the U.S. Department of Energy.  
**Why it exists:** Collects, analyzes, and publishes energy data to inform policy and markets.  
**What you’ll use:** **EIA Open Data API** (v2) for **hourly electricity demand** by balancing authority (e.g., PJM, NYIS, MISO, CISO, ERCO).  
**Access:** Free API key; JSON responses with timestamps and demand (MW).  
**Cadence/History:** Hourly; multi-year history depending on BA.  
**Pros:** National coverage, stable API, good documentation.  
**Limits:** BA-level load (not granular to every utility); occasional schema tweaks.

### PJM — PJM Interconnection (RTO)
**What it is:** The largest U.S. regional transmission organization; operates the grid and wholesale markets across 13 states + D.C.  
**Why it exists:** Ensure reliability and run competitive markets for energy, capacity, and ancillary services.  
**What you’ll use:** **PJM Data Miner 2** APIs (e.g., operational demand).  
**Access:** Free account; OAuth token; CSV/JSON downloads.  
**Cadence/History:** Hourly and sub-hourly datasets, often deep history.  
**Pros:** Rich public datasets (load, generation, outages, weather proxies).  
**Limits:** Endpoint names/fields vary by dataset; pagination and auth flow required.

### NYISO — New York Independent System Operator
**What it is:** Runs New York’s bulk power system and wholesale markets.  
**Why it exists:** Maintain reliability and facilitate competitive markets in NY.  
**What you’ll use:** Public **Actual Load** CSVs (NYCA + zones) and other reports.  
**Access:** Many CSV endpoints are open without an API key.  
**Cadence/History:** Hourly; multi-year archives.  
**Pros:** Simple CSVs; zonal + statewide totals.  
**Limits:** File layouts/links can change; you may need to merge monthly files.

### Meteostat
**What it is:** Open platform aggregating historical weather/climate from multiple networks.  
**Why it exists:** Provide easy, unified access to historical weather for research and apps.  
**What you’ll use:** **Meteostat Python library** (or REST) for **hourly** temperature, dew point, humidity, wind, and precipitation at selected stations.  
**Access:** No API key for Python; free.  
**Cadence/History:** Hourly; decades in many locations; timezone conversion handled.  
**Pros:** Rock-solid for feature engineering; painless local caching.  
**Limits:** Station coverage varies by region; occasional gaps → forward-fill/aggregation.

---

### How PowerScope uses these sources

- **Primary load input (choose one):**
  - **EIA Open Data** for fast, nationwide BA-level hourly demand.
  - **PJM Data Miner 2** for PJM-specific load (richer/varied datasets).
  - **NYISO CSVs** for an NY-focused demo (NYCA or zonal).
- **Weather features:** **Meteostat** hourly series from 2–3 nearby stations, averaged and lagged/rolled to capture thermal inertia.
- **Calendars/holidays:** Python packages (`holidays` / `workalendar`) generate business/holiday flags.

**Licensing & attribution:** Each source has its own terms (generally permissive for non-commercial use). Credit the institution if you publish results; do not imply endorsement.

**Data hygiene in PowerScope:**  
- Align everything to **hourly** cadence and a single **timezone** (configurable).  
- Detect & fill gaps conservatively; mark imputed periods.  
- Keep **train/val/test** splits strictly time-based to avoid leakage.  
- Optionally store raw pulls (`data/raw/`) and cached features (`data/processed/`) for reproducibility.

---

### Quick setup (no CSV wrangling)

- **EIA route (recommended to start):**
  1. Get a free key at api.eia.gov and set `EIA_API_KEY`.
  2. Run the included collector to write `data/raw/load.csv`.
  3. Meteostat pulls hourly weather automatically via the Python library.

- **PJM route:** Create a free PJM Data Miner account, set `PJM_CLIENT_ID`/`PJM_CLIENT_SECRET`, run the PJM collector.

- **NYISO route:** Point the NYISO collector at one or more public CSV URLs; it merges and normalizes to `timestamp,load_mw`.
