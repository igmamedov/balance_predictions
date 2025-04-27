import yaml
from datetime import timedelta
import os

def load_config(config_path=None):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str, optional): Path to the YAML configuration file.
            If None, will look for 'balance_config.yaml' in the config directory.
    
    Returns:
        dict: Configuration dictionary
    """
    if config_path is None:
        # Get the directory of this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, 'balance_config.yaml')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert retry_delay_minutes to timedelta
    config['airflow']['retry_delay'] = timedelta(minutes=config['airflow']['retry_delay_minutes'])
    del config['airflow']['retry_delay_minutes']
    
    return config

# Load default configuration
CONFIG = load_config() 