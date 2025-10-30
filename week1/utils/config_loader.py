"""
Configuration loader utility
"""

import yaml
import os
from pathlib import Path


def load_config(config_path: str = None) -> dict:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file. If None, uses default location
        
    Returns:
        dict: Configuration dictionary
    """
    if config_path is None:
        # Default to configs/config.yaml in the week1 directory
        config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directories if they don't exist
    for dir_key in ['results_dir', 'plots_dir', 'logs_dir']:
        if dir_key in config.get('output', {}):
            dir_path = config['output'][dir_key]
            os.makedirs(dir_path, exist_ok=True)
    
    return config


def get_config_value(config: dict, key_path: str, default=None):
    """
    Get nested configuration value using dot notation
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated path (e.g., 'model.name')
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    keys = key_path.split('.')
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value

