"""
Input/Output utilities for PowerScope.
"""

import pandas as pd
import numpy as np
import yaml
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Union, Optional
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    logger.info(f"Loaded configuration from {config_path}")
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False, indent=2)
        
    logger.info(f"Saved configuration to {config_path}")


def load_csv(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Load CSV file with error handling.
    
    Args:
        file_path: Path to CSV file
        **kwargs: Additional arguments for pd.read_csv
        
    Returns:
        DataFrame
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")
        
    try:
        df = pd.read_csv(file_path, **kwargs)
        logger.info(f"Loaded CSV with shape {df.shape} from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading CSV from {file_path}: {e}")
        raise


def save_csv(df: pd.DataFrame, file_path: str, **kwargs) -> None:
    """
    Save DataFrame to CSV with directory creation.
    
    Args:
        df: DataFrame to save
        file_path: Path to save CSV
        **kwargs: Additional arguments for to_csv
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(file_path, index=False, **kwargs)
    logger.info(f"Saved CSV with shape {df.shape} to {file_path}")


def load_parquet(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Load Parquet file with error handling.
    
    Args:
        file_path: Path to Parquet file
        **kwargs: Additional arguments for pd.read_parquet
        
    Returns:
        DataFrame
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {file_path}")
        
    try:
        df = pd.read_parquet(file_path, **kwargs)
        logger.info(f"Loaded Parquet with shape {df.shape} from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading Parquet from {file_path}: {e}")
        raise


def save_parquet(df: pd.DataFrame, file_path: str, **kwargs) -> None:
    """
    Save DataFrame to Parquet with directory creation.
    
    Args:
        df: DataFrame to save
        file_path: Path to save Parquet
        **kwargs: Additional arguments for to_parquet
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(file_path, index=False, **kwargs)
    logger.info(f"Saved Parquet with shape {df.shape} to {file_path}")


def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Dictionary from JSON
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
        
    with open(file_path, 'r') as f:
        data = json.load(f)
        
    logger.info(f"Loaded JSON from {file_path}")
    return data


def save_json(data: Dict[str, Any], file_path: str, indent: int = 2) -> None:
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        file_path: Path to save JSON
        indent: JSON indentation
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=indent, default=str)
        
    logger.info(f"Saved JSON to {file_path}")


def load_pickle(file_path: str) -> Any:
    """
    Load pickle file.
    
    Args:
        file_path: Path to pickle file
        
    Returns:
        Unpickled object
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {file_path}")
        
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        
    logger.info(f"Loaded pickle from {file_path}")
    return data


def save_pickle(obj: Any, file_path: str) -> None:
    """
    Save object to pickle file.
    
    Args:
        obj: Object to save
        file_path: Path to save pickle
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)
        
    logger.info(f"Saved pickle to {file_path}")


def ensure_directory(dir_path: str) -> Path:
    """
    Ensure directory exists, create if not.
    
    Args:
        dir_path: Directory path
        
    Returns:
        Path object
    """
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def list_files(directory: str, 
               pattern: str = "*",
               recursive: bool = False) -> list:
    """
    List files in directory with pattern matching.
    
    Args:
        directory: Directory to search
        pattern: File pattern (e.g., "*.csv")
        recursive: Search recursively
        
    Returns:
        List of file paths
    """
    directory = Path(directory)
    
    if not directory.exists():
        return []
        
    if recursive:
        files = list(directory.rglob(pattern))
    else:
        files = list(directory.glob(pattern))
        
    return [f for f in files if f.is_file()]


def get_file_size(file_path: str) -> int:
    """
    Get file size in bytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in bytes
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        return 0
        
    return file_path.stat().st_size


def copy_file(src: str, dst: str) -> None:
    """
    Copy file from source to destination.
    
    Args:
        src: Source file path
        dst: Destination file path
    """
    import shutil
    
    src_path = Path(src)
    dst_path = Path(dst)
    
    if not src_path.exists():
        raise FileNotFoundError(f"Source file not found: {src_path}")
        
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)
    
    logger.info(f"Copied {src_path} to {dst_path}")


def validate_file_extension(file_path: str, 
                          expected_extensions: list) -> bool:
    """
    Validate file extension.
    
    Args:
        file_path: Path to file
        expected_extensions: List of expected extensions (e.g., ['.csv', '.parquet'])
        
    Returns:
        True if extension is valid
    """
    file_path = Path(file_path)
    return file_path.suffix.lower() in [ext.lower() for ext in expected_extensions]


class DataFrameInfo:
    """Utility class for DataFrame information and validation."""
    
    @staticmethod
    def get_info(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive DataFrame information.
        
        Args:
            df: DataFrame
            
        Returns:
            Dictionary with DataFrame info
        """
        return {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).to_dict(),
            'null_counts': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
            'datetime_columns': df.select_dtypes(include=['datetime64']).columns.tolist()
        }
        
    @staticmethod
    def validate_required_columns(df: pd.DataFrame, 
                                required_columns: list) -> bool:
        """
        Validate that DataFrame has required columns.
        
        Args:
            df: DataFrame
            required_columns: List of required column names
            
        Returns:
            True if all required columns exist
        """
        missing_columns = set(required_columns) - set(df.columns)
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
            
        return True