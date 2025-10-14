"""Configuration loader utility."""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "configs/training_config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def update_config(config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Update configuration with keyword arguments.
    
    Args:
        config: Base configuration dictionary
        **kwargs: Key-value pairs to update
        
    Returns:
        Updated configuration dictionary
    """
    for key, value in kwargs.items():
        if '.' in key:
            # Handle nested keys like 'agent.learning_rate'
            parts = key.split('.')
            current = config
            for part in parts[:-1]:
                current = current.setdefault(part, {})
            current[parts[-1]] = value
        else:
            config[key] = value
    
    return config
