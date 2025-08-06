"""
Warning suppression utilities for cleaner console output.
Suppresses known warnings from CUDA/Triton fallbacks.
"""
import warnings
import logging
import os

def suppress_cuda_warnings():
    """Suppress common CUDA/Triton warnings when falling back to CPU."""
    # Suppress Triton kernel warnings
    warnings.filterwarnings("ignore", message=".*Failed to launch Triton kernels.*")
    warnings.filterwarnings("ignore", message=".*falling back to.*DTW.*")
    warnings.filterwarnings("ignore", message=".*falling back to.*median kernel.*")
    
    # Suppress torch warnings about CPU fallback
    warnings.filterwarnings("ignore", message=".*torch.distributed.checkpoint.*")
    warnings.filterwarnings("ignore", message=".*Using torch.backends.cuda.matmul.*")
    
    # Set environment variables to reduce CUDA initialization messages
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
    
    # Configure logging to reduce noise
    logging.getLogger("whisper").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)

def configure_cpu_mode():
    """Configure the application to run efficiently in CPU mode."""
    # Force PyTorch to use CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # Set optimal CPU settings
    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
    os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())
    
    # Suppress CUDA-related warnings
    suppress_cuda_warnings()

def smart_device_setup():
    """Intelligently configure device usage with minimal warnings."""
    import torch
    
    # Suppress warnings during device detection
    suppress_cuda_warnings()
    
    # Check CUDA availability without triggering warnings
    try:
        if torch.cuda.is_available():
            device = "cuda"
            device_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU detected: {device_name} ({memory_gb:.1f}GB)")
        else:
            device = "cpu"
            cpu_count = os.cpu_count()
            print(f"Using CPU mode: {cpu_count} cores available")
            configure_cpu_mode()
    except Exception:
        # Fallback to CPU if any CUDA detection fails
        device = "cpu"
        configure_cpu_mode()
        print("Fallback to CPU mode for compatibility")
    
    return device