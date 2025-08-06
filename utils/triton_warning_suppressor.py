"""
Triton kernel warning suppressor for RTX 3060 systems.
Suppresses cosmetic Triton warnings that don't affect processing.
"""
import warnings
import logging
from functools import wraps

def suppress_triton_warnings():
    """Suppress Triton kernel warnings that are cosmetic on RTX 3060."""
    warnings.filterwarnings("ignore", message="Failed to launch Triton kernels")
    warnings.filterwarnings("ignore", message="falling back to a slower median kernel")
    warnings.filterwarnings("ignore", message="falling back to a slower DTW implementation")
    
    # Suppress specific Whisper timing warnings
    logging.getLogger("whisper.timing").setLevel(logging.ERROR)

def quiet_transcription(func):
    """Decorator to suppress Triton warnings during transcription."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        suppress_triton_warnings()
        return func(*args, **kwargs)
    return wrapper