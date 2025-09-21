"""
Weather data from Meteostat API.
"""

import pandas as pd
from typing import List, Tuple, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class WeatherMeteostatLoader:
    """Load weather data from Meteostat."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize weather data loader.
        
        Args:
            api_key: Meteostat API key (if required)
        """
        self.api_key = api_key
        
    def load_weather_data(self,
                         stations: List[str],
                         start_date: str,
                         end_date: str) -> pd.DataFrame:
        """
        Load weather data for given stations and date range.
        
        Args:
            stations: List of weather station IDs
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with weather data
        """
        try:
            from meteostat import Point, Hourly
            
            # Convert dates
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            
            weather_dfs = []
            
            for station in stations:
                try:
                    # Create Point object (assuming station is lat,lon format)
                    if ',' in station:
                        lat, lon = map(float, station.split(','))
                        location = Point(lat, lon)
                        
                        # Get hourly data
                        data = Hourly(location, start, end)
                        df = data.fetch()
                        
                        if not df.empty:
                            df['station'] = station
                            df = df.reset_index()
                            weather_dfs.append(df)
                            
                except Exception as e:
                    logger.error(f"Error loading data for station {station}: {e}")
                    
            if weather_dfs:
                return pd.concat(weather_dfs, ignore_index=True)
            else:
                logger.warning("No weather data loaded")
                return pd.DataFrame()
                
        except ImportError:
            logger.error("meteostat package not installed. Install with: pip install meteostat")
            return pd.DataFrame()
            
    def get_temperature_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temperature-based features.
        
        Args:
            df: Weather DataFrame
            
        Returns:
            DataFrame with temperature features
        """
        if df.empty or 'temp' not in df.columns:
            return df
            
        # Calculate temperature features
        df['temp_rolling_24h'] = df['temp'].rolling(window=24, min_periods=1).mean()
        df['temp_max_24h'] = df['temp'].rolling(window=24, min_periods=1).max()
        df['temp_min_24h'] = df['temp'].rolling(window=24, min_periods=1).min()
        
        # Cooling/heating degree days (base 65°F)
        base_temp = 65
        df['cdd'] = (df['temp'] - base_temp).clip(lower=0)  # Cooling degree
        df['hdd'] = (base_temp - df['temp']).clip(lower=0)  # Heating degree
        
        return df
        
    def merge_with_demand(self, 
                         demand_df: pd.DataFrame, 
                         weather_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge weather data with demand data.
        
        Args:
            demand_df: Electricity demand DataFrame
            weather_df: Weather DataFrame
            
        Returns:
            Merged DataFrame
        """
        if weather_df.empty:
            return demand_df
            
        # Ensure timestamp columns
        if 'time' in weather_df.columns:
            weather_df = weather_df.rename(columns={'time': 'timestamp'})
            
        # Convert to datetime if needed
        for df in [demand_df, weather_df]:
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
        # Merge on timestamp
        merged = pd.merge(demand_df, weather_df, on='timestamp', how='left')
        
        return merged