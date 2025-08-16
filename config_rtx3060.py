"""
RTX 3060 optimized configuration for video summarization.
No CPU fallback - pure CUDA acceleration.
"""
import os
from pathlib import Path

# Application settings
APP_NAME = "Video Summarization Engine - RTX 3060"
VERSION = "2.0.0-RTX3060"

# Directory paths
ROOT_DIR = Path(__file__).parent
MODELS_DIR = ROOT_DIR / "models"
DATABASE_PATH = ROOT_DIR / "video_processing.db"
OUTPUT_DIR = ROOT_DIR / "output"
TEMP_DIR = ROOT_DIR / "temp"

# Video processing settings
SUPPORTED_FORMATS = ['.mp4', '.mkv', '.avi', '.mov', '.wmv']
SUBTITLE_FORMATS = ['.srt', '.vtt', '.ass', '.ssa']

# RTX 3060 GPU settings (12.9GB VRAM)
CUDA_DEVICE = "cuda:0"
MAX_GPU_MEMORY_GB = 11.0  # Reserve 1.9GB for system
GPU_MEMORY_FRACTION = 0.85  # Use 85% of available VRAM
ENABLE_MIXED_PRECISION = True  # RTX 3060 supports Tensor Cores
ENABLE_TF32 = True  # RTX 3060 optimization

# Scene detection settings - SPEED OPTIMIZED
SCENE_DETECTION_THRESHOLD = 35.0  # Higher = fewer scenes = faster
MIN_SCENE_LENGTH_SECONDS = 3.0  # Longer minimum = faster processing
ADAPTIVE_THRESHOLD = 3.0
FRAME_SKIP = 2  # Process every 2nd frame for 2x speed

# RTX 3060 optimized Whisper settings - SPEED OPTIMIZED
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")  # 4x faster than large, 95%+ accuracy
WHISPER_LANGUAGE = "en"
CHUNK_LENGTH_MS = 20000  # Reduced to 20 seconds for faster processing
WHISPER_DEVICE = "cuda"
WHISPER_BATCH_SIZE = 12  # Increased for speed (from 8)

# LLM settings - RTX 3060 optimized - Conservative for stability
LLM_MODEL_PATH = MODELS_DIR / "local_llm"
LLM_MAX_TOKENS = 2048  # Reduced from 4096 for memory efficiency
LLM_TEMPERATURE = 0.7
LLM_DEVICE = "cuda"
LLM_BATCH_SIZE = 4  # Reduced from 8 for better memory management

# Summary settings
SUMMARY_LENGTH_PERCENT = 15  # Target 15% of original length
MAX_SUMMARY_LENGTH_MINUTES = 25  # Increased for RTX 3060 performance
MIN_SUMMARY_LENGTH_MINUTES = 2

# Validation settings
VALIDATION_METRICS = ["f1_score", "precision", "recall"]
BENCHMARK_DATASETS = ["tvsum", "summe"]

# RTX 3060 batch processing - SPEED OPTIMIZED
DEFAULT_BATCH_SIZE = 1  # Single video for max speed and memory efficiency  
MAX_BATCH_SIZE = 2  # Maximum for complex videos
PROCESSING_TIMEOUT_HOURS = 2  # Faster expected completion
CLEANUP_FREQUENCY = 3  # Clean up every 3 videos instead of every video

# RTX 3060 Performance Optimization
CUDA_BENCHMARK = True  # Enable cuDNN benchmark
CUDA_DETERMINISTIC = False  # Disable for better performance
MEMORY_EFFICIENT = True  # Enable gradient checkpointing

# Development/Testing Configuration
DISABLE_AUTO_DOWNLOADS = os.getenv('DISABLE_AUTO_DOWNLOADS', 'false').lower() == 'true'
DISABLE_DATABASE_INIT = os.getenv('DISABLE_DATABASE_INIT', 'false').lower() == 'true'
OFFLINE_MODE = os.getenv('OFFLINE_MODE', 'false').lower() == 'true'

# Create necessary directories
for directory in [MODELS_DIR, OUTPUT_DIR, TEMP_DIR]:
    directory.mkdir(exist_ok=True)

# RTX 3060 CUDA Initialization
import torch
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = CUDA_BENCHMARK
    torch.backends.cudnn.deterministic = CUDA_DETERMINISTIC
    torch.backends.cuda.matmul.allow_tf32 = ENABLE_TF32
    if ENABLE_MIXED_PRECISION:
        torch.backends.cudnn.allow_tf32 = True