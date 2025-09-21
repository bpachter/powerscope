"""
ISO local CSV data loading and processing.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ISOLocalCSVLoader:
    """Load and process ISO local CSV data."""
    
    def __init__(self, data_path: str):
        """
        Initialize ISO CSV loader.
        
        Args:
            data_path: Path to ISO CSV data directory
        """
        self.data_path = Path(data_path)
        
    def load_demand_data(self, 
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load electricity demand data from CSV files.
        
        Args:
            start_date: Start date for data loading (YYYY-MM-DD)
            end_date: End date for data loading (YYYY-MM-DD)
            
        Returns:
            DataFrame with demand data
        """
        csv_files = list(self.data_path.glob("*.csv"))
        
        if not csv_files:
            logger.warning(f"No CSV files found in {self.data_path}")
            return pd.DataFrame()
            
        dfs = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                dfs.append(df)
            except Exception as e:
                logger.error(f"Error loading {csv_file}: {e}")
                
        if not dfs:
            return pd.DataFrame()
            
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df = combined_df.sort_values('timestamp')
        
        # Filter by date range if provided
        if start_date:
            combined_df = combined_df[combined_df['timestamp'] >= start_date]
        if end_date:
            combined_df = combined_df[combined_df['timestamp'] <= end_date]
            
        return combined_df
        
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the loaded data."""
        df = self.load_demand_data()
        
        if df.empty:
            return {}
            
        return {
            'total_records': len(df),
            'date_range': {
                'start': df['timestamp'].min(),
                'end': df['timestamp'].max()
            },
            'columns': df.columns.tolist(),
            'missing_values': df.isnull().sum().to_dict()
        }