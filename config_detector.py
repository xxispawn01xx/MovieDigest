"""
Smart configuration detector for video summarization.
Automatically detects environment and applies appropriate config.
"""
import torch
import os
from pathlib import Path

def detect_environment():
    """Detect if running in Replit cloud or local RTX 3060 environment."""
    # Check for Replit environment indicators
    is_replit = (
        os.getenv('REPLIT_ENVIRONMENT') is not None or
        os.getenv('REPL_SLUG') is not None or
        'replit' in str(Path.cwd()).lower() or
        not torch.cuda.is_available()
    )
    
    if is_replit:
        return "replit_cloud"
    elif torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0).lower()
        if "rtx 3060" in gpu_name or "geforce rtx 3060" in gpu_name:
            return "rtx_3060_local"
        else:
            return "cuda_generic"
    else:
        return "cpu_only"

def get_config():
    """Get appropriate configuration based on environment."""
    env = detect_environment()
    
    if env == "rtx_3060_local":
        # Import RTX 3060 optimized config
        import config_rtx3060 as config
        print("RTX 3060 configuration loaded")
    else:
        # Import standard config with CPU fallback
        import config as config
        print(f"Standard configuration loaded for {env}")
    
    return config, env

# Initialize configuration
CONFIG, ENVIRONMENT = get_config()