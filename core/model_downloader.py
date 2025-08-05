"""
Model downloader for managing Whisper and LLM models locally.
Handles downloading, caching, and model management.
"""
import os
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
import logging
import json
import config

logger = logging.getLogger(__name__)

class ModelDownloader:
    """Manages downloading and caching of AI models for offline use."""
    
    def __init__(self, models_dir: str = "models"):
        """Initialize model downloader."""
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Whisper models directory
        self.whisper_dir = self.models_dir / "whisper"
        self.whisper_dir.mkdir(exist_ok=True)
        
        # LLM models directory
        self.llm_dir = self.models_dir / "llm"
        self.llm_dir.mkdir(exist_ok=True)
        
        # Model catalog
        self.whisper_models = {
            'tiny': 'tiny model (1.5x speed, lower accuracy)',
            'base': 'base model (balanced speed/accuracy)',
            'small': 'small model (good accuracy)',
            'medium': 'medium model (better accuracy, slower)',
            'large': 'large model (best accuracy, slowest)',
            'large-v3': 'large-v3 model (latest, best quality)'
        }
    
    def list_available_whisper_models(self) -> Dict[str, str]:
        """Get list of available Whisper models."""
        return self.whisper_models.copy()
    
    def list_downloaded_whisper_models(self) -> List[str]:
        """Get list of already downloaded Whisper models."""
        downloaded = []
        
        # Check for Whisper cache directory
        whisper_cache = Path.home() / ".cache" / "whisper"
        if whisper_cache.exists():
            for model_file in whisper_cache.glob("*.pt"):
                model_name = model_file.stem
                if model_name in self.whisper_models:
                    downloaded.append(model_name)
        
        return downloaded
    
    def download_whisper_model(self, model_name: str) -> bool:
        """
        Download a Whisper model for offline use.
        
        Args:
            model_name: Name of the Whisper model to download
            
        Returns:
            True if download successful
        """
        if config.DISABLE_AUTO_DOWNLOADS or config.OFFLINE_MODE:
            logger.info(f"Auto-downloads disabled, skipping {model_name}")
            return False
            
        if model_name not in self.whisper_models:
            logger.error(f"Unknown Whisper model: {model_name}")
            return False
        
        try:
            logger.info(f"Downloading Whisper model: {model_name}")
            
            # Use whisper command to download model
            import whisper
            model = whisper.load_model(model_name)
            
            logger.info(f"Whisper model '{model_name}' downloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download Whisper model '{model_name}': {e}")
            return False
    
    def get_whisper_model_info(self, model_name: str) -> Optional[Dict]:
        """
        Get information about a Whisper model.
        
        Args:
            model_name: Name of the Whisper model
            
        Returns:
            Dictionary with model information
        """
        if model_name not in self.whisper_models:
            return None
        
        # Check if model is downloaded
        whisper_cache = Path.home() / ".cache" / "whisper"
        model_file = whisper_cache / f"{model_name}.pt"
        
        info = {
            'name': model_name,
            'description': self.whisper_models[model_name],
            'downloaded': model_file.exists(),
            'file_path': str(model_file) if model_file.exists() else None,
            'size_mb': model_file.stat().st_size / (1024 * 1024) if model_file.exists() else None
        }
        
        return info
    
    def suggest_whisper_model(self, video_duration_minutes: float, 
                            quality_preference: str = "balanced") -> str:
        """
        Suggest optimal Whisper model based on video length and quality preference.
        
        Args:
            video_duration_minutes: Length of video in minutes
            quality_preference: "speed", "balanced", or "quality"
            
        Returns:
            Recommended model name
        """
        if quality_preference == "speed":
            if video_duration_minutes > 120:  # 2+ hours
                return "tiny"
            elif video_duration_minutes > 60:  # 1+ hour
                return "base"
            else:
                return "small"
        
        elif quality_preference == "quality":
            if video_duration_minutes > 180:  # 3+ hours
                return "medium"
            else:
                return "large-v3"
        
        else:  # balanced
            if video_duration_minutes > 150:  # 2.5+ hours
                return "base"
            elif video_duration_minutes > 90:  # 1.5+ hours
                return "small"
            else:
                return "medium"
    
    def create_model_info_file(self) -> str:
        """
        Create a JSON file with information about all available models.
        
        Returns:
            Path to the created info file
        """
        info_file = self.models_dir / "model_info.json"
        
        # Gather Whisper model info
        whisper_info = {}
        for model_name in self.whisper_models:
            whisper_info[model_name] = self.get_whisper_model_info(model_name)
        
        # Create complete info structure
        model_info = {
            'whisper_models': whisper_info,
            'llm_models': {
                'status': 'Place LLM models in models/llm/ directory',
                'supported_formats': ['transformers', 'gguf', 'safetensors'],
                'recommended_models': [
                    'microsoft/DialoGPT-medium',
                    'microsoft/DialoGPT-large', 
                    'gpt2-medium',
                    'distilgpt2'
                ]
            },
            'last_updated': str(Path(__file__).stat().st_mtime)
        }
        
        with open(info_file, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info(f"Model info file created: {info_file}")
        return str(info_file)
    
    def check_model_requirements(self) -> Dict[str, bool]:
        """
        Check if minimum model requirements are met.
        
        Returns:
            Dictionary with requirement status
        """
        requirements = {
            'whisper_available': False,
            'whisper_model_downloaded': False,
            'transformers_available': False,
            'llm_model_available': False,
            'storage_space_ok': True
        }
        
        # Check Whisper availability
        try:
            import whisper
            requirements['whisper_available'] = True
            
            # Check if any Whisper model is downloaded
            downloaded_models = self.list_downloaded_whisper_models()
            requirements['whisper_model_downloaded'] = len(downloaded_models) > 0
            
        except ImportError:
            pass
        
        # Check transformers availability
        try:
            import transformers
            requirements['transformers_available'] = True
        except ImportError:
            pass
        
        # Check for LLM models
        if self.llm_dir.exists():
            llm_files = list(self.llm_dir.rglob("*.bin")) + list(self.llm_dir.rglob("*.safetensors"))
            requirements['llm_model_available'] = len(llm_files) > 0
        
        # Check storage space (require at least 2GB free)
        import shutil
        free_space_gb = shutil.disk_usage(self.models_dir).free / (1024**3)
        requirements['storage_space_ok'] = free_space_gb > 2.0
        
        return requirements