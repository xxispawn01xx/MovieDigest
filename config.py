"""
Configuration settings for the video summarization application.
Centralized configuration management following coursequery patterns.
"""
import os
from pathlib import Path

# Application settings
APP_NAME = "Video Summarization Engine"
VERSION = "1.0.0"

# Directory paths
ROOT_DIR = Path(__file__).parent
MODELS_DIR = ROOT_DIR / "models"
DATABASE_PATH = ROOT_DIR / "video_processing.db"
OUTPUT_DIR = ROOT_DIR / "output"
TEMP_DIR = ROOT_DIR / "temp"

# Video processing settings
SUPPORTED_FORMATS = ['.mp4', '.mkv', '.avi', '.mov', '.wmv']
SUBTITLE_FORMATS = ['.srt', '.vtt', '.ass', '.ssa']

# GPU settings
CUDA_DEVICE = "cuda:0" if os.getenv("CUDA_VISIBLE_DEVICES") else "cuda"
MAX_GPU_MEMORY_GB = 10  # Reserve 2GB for system on RTX 3060 12GB

# Scene detection settings
SCENE_DETECTION_THRESHOLD = 30.0
MIN_SCENE_LENGTH_SECONDS = 2.0
ADAPTIVE_THRESHOLD = 3.0

# Transcription settings
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")  # base, small, medium, large
WHISPER_LANGUAGE = "en"
CHUNK_LENGTH_MS = 30000  # 30 seconds

# LLM settings
LLM_MODEL_PATH = MODELS_DIR / "local_llm"
LLM_MAX_TOKENS = 2048
LLM_TEMPERATURE = 0.7

# Summary settings
SUMMARY_LENGTH_PERCENT = 15  # Target 15% of original length
MAX_SUMMARY_LENGTH_MINUTES = 20
MIN_SUMMARY_LENGTH_MINUTES = 2

# Validation settings
VALIDATION_METRICS = ["f1_score", "precision", "recall"]
BENCHMARK_DATASETS = ["tvsum", "summe"]

# Batch processing
DEFAULT_BATCH_SIZE = 2
MAX_BATCH_SIZE = 5
PROCESSING_TIMEOUT_HOURS = 6

# Development/Testing Configuration
DISABLE_AUTO_DOWNLOADS = os.getenv('DISABLE_AUTO_DOWNLOADS', 'false').lower() == 'true'
DISABLE_DATABASE_INIT = os.getenv('DISABLE_DATABASE_INIT', 'false').lower() == 'true'
OFFLINE_MODE = os.getenv('OFFLINE_MODE', 'false').lower() == 'true'

# Create necessary directories
for directory in [MODELS_DIR, OUTPUT_DIR, TEMP_DIR]:
    directory.mkdir(exist_ok=True)
