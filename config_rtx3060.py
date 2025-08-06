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

# Scene detection settings
SCENE_DETECTION_THRESHOLD = 30.0
MIN_SCENE_LENGTH_SECONDS = 2.0
ADAPTIVE_THRESHOLD = 3.0

# RTX 3060 optimized Whisper settings
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "large")  # RTX 3060 can handle large
WHISPER_LANGUAGE = "en"
CHUNK_LENGTH_MS = 30000  # 30 seconds
WHISPER_DEVICE = "cuda"
WHISPER_BATCH_SIZE = 16  # RTX 3060 optimized batch size

# LLM settings - RTX 3060 optimized
LLM_MODEL_PATH = MODELS_DIR / "local_llm"
LLM_MAX_TOKENS = 4096  # Increased for RTX 3060
LLM_TEMPERATURE = 0.7
LLM_DEVICE = "cuda"
LLM_BATCH_SIZE = 8  # RTX 3060 optimized

# Summary settings
SUMMARY_LENGTH_PERCENT = 15  # Target 15% of original length
MAX_SUMMARY_LENGTH_MINUTES = 25  # Increased for RTX 3060 performance
MIN_SUMMARY_LENGTH_MINUTES = 2

# Validation settings
VALIDATION_METRICS = ["f1_score", "precision", "recall"]
BENCHMARK_DATASETS = ["tvsum", "summe"]

# RTX 3060 optimized batch processing
DEFAULT_BATCH_SIZE = 3  # RTX 3060 can handle 3 videos
MAX_BATCH_SIZE = 4  # Maximum for 12.9GB VRAM
PROCESSING_TIMEOUT_HOURS = 4  # Faster with RTX 3060

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