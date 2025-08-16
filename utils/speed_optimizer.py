"""
Speed optimization utilities for video processing.
Provides fast processing modes and performance monitoring.
"""

import logging
import time
import torch
from typing import Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class SpeedOptimizer:
    """Optimizes video processing for speed over accuracy."""
    
    def __init__(self):
        self.optimization_level = "standard"  # standard, fast, ultra_fast
        self.speed_metrics = {}
        
    def set_optimization_level(self, level: str):
        """Set optimization level: standard, fast, ultra_fast"""
        valid_levels = ["standard", "fast", "ultra_fast"]
        if level not in valid_levels:
            raise ValueError(f"Invalid optimization level. Choose from: {valid_levels}")
        self.optimization_level = level
        logger.info(f"Speed optimization set to: {level}")
    
    def get_optimized_whisper_config(self) -> Dict:
        """Get Whisper configuration based on optimization level."""
        base_config = {
            'device': 'cuda',
            'compute_type': 'float16',
            'language': 'en',
            'condition_on_previous_text': False,
            'no_speech_threshold': 0.6,
            'logprob_threshold': -1.0,
            'compression_ratio_threshold': 2.4,
            'word_timestamps': False,  # Disable for speed
        }
        
        if self.optimization_level == "standard":
            base_config.update({
                'model': 'base',
                'batch_size': 8,
                'chunk_length': 20,
                'beam_size': 1,  # Greedy decoding for speed
            })
        elif self.optimization_level == "fast":
            base_config.update({
                'model': 'base', 
                'batch_size': 12,
                'chunk_length': 15,
                'beam_size': 1,
                'no_speech_threshold': 0.8,  # Skip more silence
            })
        elif self.optimization_level == "ultra_fast":
            base_config.update({
                'model': 'tiny',  # Fastest model
                'batch_size': 16,
                'chunk_length': 10,
                'beam_size': 1,
                'no_speech_threshold': 0.9,  # Skip most silence
                'logprob_threshold': -0.5,  # Less conservative
            })
        
        return base_config
    
    def get_optimized_scene_config(self) -> Dict:
        """Get scene detection configuration based on optimization level."""
        base_config = {
            'detector_type': 'ContentDetector',
            'adaptive_threshold': False,  # Disable for speed
            'luma_only': True,  # Only process brightness channel
        }
        
        if self.optimization_level == "standard":
            base_config.update({
                'threshold': 32.0,
                'min_scene_len': 3.0,
                'frame_skip': 1,
            })
        elif self.optimization_level == "fast":
            base_config.update({
                'threshold': 35.0,  # Higher = fewer scenes
                'min_scene_len': 4.0,
                'frame_skip': 2,  # Every 2nd frame
            })
        elif self.optimization_level == "ultra_fast":
            base_config.update({
                'threshold': 40.0,  # Much higher threshold
                'min_scene_len': 5.0,
                'frame_skip': 3,  # Every 3rd frame
            })
        
        return base_config
    
    def optimize_gpu_settings(self):
        """Apply GPU optimizations for speed."""
        if torch.cuda.is_available():
            # Enable optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Set memory management
            torch.cuda.empty_cache()
            
            logger.info("Applied GPU speed optimizations")
    
    def estimate_processing_time(self, video_duration_minutes: float, file_size_gb: float) -> Dict:
        """Estimate processing time based on optimization level."""
        
        # Base processing rates (minutes of video per minute of processing time)
        base_rates = {
            "standard": 8,    # 1 hour video = 7.5 minutes processing
            "fast": 15,       # 1 hour video = 4 minutes processing  
            "ultra_fast": 25  # 1 hour video = 2.4 minutes processing
        }
        
        rate = base_rates[self.optimization_level]
        
        # Adjust for file size (larger files take longer)
        size_multiplier = 1.0 + (file_size_gb - 1.0) * 0.1 if file_size_gb > 1.0 else 1.0
        
        estimated_minutes = (video_duration_minutes / rate) * size_multiplier
        
        return {
            'estimated_minutes': estimated_minutes,
            'estimated_seconds': estimated_minutes * 60,
            'optimization_level': self.optimization_level,
            'processing_rate': f"{rate}x real-time",
            'file_size_factor': size_multiplier
        }
    
    def track_processing_speed(self, video_path: str, start_time: float, end_time: float):
        """Track actual processing speed for metrics."""
        duration = end_time - start_time
        video_file = Path(video_path)
        
        # Get video duration (would need to implement video duration detection)
        # For now, estimate based on file size
        file_size_gb = video_file.stat().st_size / (1024**3)
        
        self.speed_metrics[video_path] = {
            'processing_time_seconds': duration,
            'processing_time_minutes': duration / 60,
            'file_size_gb': file_size_gb,
            'optimization_level': self.optimization_level,
            'timestamp': time.time()
        }
        
        logger.info(f"Video processed in {duration/60:.1f} minutes (level: {self.optimization_level})")
    
    def get_speed_recommendations(self, current_processing_time_hours: float) -> Dict:
        """Get recommendations to improve processing speed."""
        recommendations = []
        
        if current_processing_time_hours > 2:
            recommendations.extend([
                "Switch to 'base' Whisper model (4x faster than 'large')",
                "Enable frame skipping in scene detection",
                "Increase batch sizes for GPU utilization",
                "Use ultra_fast optimization mode"
            ])
        elif current_processing_time_hours > 1:
            recommendations.extend([
                "Consider 'fast' optimization mode",
                "Increase Whisper batch size",
                "Skip detailed narrative analysis for speed"
            ])
        else:
            recommendations.append("Processing speed is already optimized")
        
        return {
            'current_time_hours': current_processing_time_hours,
            'recommendations': recommendations,
            'expected_speedup': "3-8x faster" if current_processing_time_hours > 2 else "1.5-3x faster"
        }

# Global optimizer instance
speed_optimizer = SpeedOptimizer()

def apply_speed_optimizations(level: str = "fast"):
    """Apply speed optimizations at the specified level."""
    speed_optimizer.set_optimization_level(level)
    speed_optimizer.optimize_gpu_settings()
    
    logger.info(f"Speed optimizations applied at '{level}' level")
    return speed_optimizer

def get_fast_whisper_config():
    """Get fast Whisper configuration."""
    return speed_optimizer.get_optimized_whisper_config()

def get_fast_scene_config():
    """Get fast scene detection configuration.""" 
    return speed_optimizer.get_optimized_scene_config()