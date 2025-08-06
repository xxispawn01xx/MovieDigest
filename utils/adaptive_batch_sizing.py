"""
Adaptive batch sizing utility for dynamic memory-based batch size optimization.
Automatically adjusts batch sizes based on available GPU memory and video characteristics.
"""
import torch
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import cv2

from config_detector import CONFIG as config

logger = logging.getLogger(__name__)

class AdaptiveBatchSizer:
    """Dynamically adjusts batch sizes based on memory availability and video characteristics."""
    
    def __init__(self):
        """Initialize adaptive batch sizer."""
        self.cuda_available = torch.cuda.is_available()
        self.memory_history = []
        self.batch_performance = {}
        
    def get_video_memory_estimate(self, video_path: Path) -> float:
        """
        Estimate GPU memory required for processing a video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Estimated memory in GB
        """
        try:
            # Get video properties
            cap = cv2.VideoCapture(str(video_path))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            duration_minutes = (frame_count / fps) / 60 if fps > 0 else 60
            resolution_factor = (width * height) / (1920 * 1080)  # Normalize to 1080p
            
            # Base memory estimates (empirically derived)
            base_memory = 2.0  # Base model memory
            whisper_memory = min(duration_minutes * 0.1, 3.0)  # ~0.1GB per minute, cap at 3GB
            scene_memory = frame_count * resolution_factor * 0.000001  # Scene detection
            
            total_estimate = base_memory + whisper_memory + scene_memory
            
            logger.debug(f"Memory estimate for {video_path.name}: {total_estimate:.2f}GB "
                        f"(duration: {duration_minutes:.1f}min, resolution: {width}x{height})")
            
            return total_estimate
            
        except Exception as e:
            logger.warning(f"Could not estimate memory for {video_path}: {e}")
            # Conservative fallback estimate
            return 4.0
    
    def get_optimal_batch_size(self, video_paths: list, target_memory_gb: float = 9.0) -> int:
        """
        Calculate optimal batch size based on available memory and video characteristics.
        
        Args:
            video_paths: List of video file paths
            target_memory_gb: Target memory usage in GB
            
        Returns:
            Optimal batch size
        """
        if not self.cuda_available or not video_paths:
            return 1
        
        try:
            # Get current memory usage
            allocated = torch.cuda.memory_allocated() / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            available = total - allocated
            
            # Use conservative target (leave buffer)
            usable_memory = min(available * 0.8, target_memory_gb)
            
            # Estimate memory per video
            memory_estimates = []
            for video_path in video_paths[:10]:  # Check first 10 for estimation
                estimate = self.get_video_memory_estimate(Path(video_path))
                memory_estimates.append(estimate)
            
            if not memory_estimates:
                return 1
            
            # Use average estimate for batch sizing
            avg_memory_per_video = sum(memory_estimates) / len(memory_estimates)
            
            # Calculate batch size
            optimal_batch_size = max(1, int(usable_memory / avg_memory_per_video))
            
            # Apply constraints
            optimal_batch_size = min(optimal_batch_size, config.MAX_BATCH_SIZE)
            optimal_batch_size = min(optimal_batch_size, len(video_paths))
            
            logger.info(f"Optimal batch size: {optimal_batch_size} "
                       f"(available: {available:.1f}GB, avg per video: {avg_memory_per_video:.1f}GB)")
            
            return optimal_batch_size
            
        except Exception as e:
            logger.error(f"Batch size calculation failed: {e}")
            return config.DEFAULT_BATCH_SIZE
    
    def get_memory_conservative_settings(self) -> Dict:
        """
        Get memory-conservative processing settings.
        
        Returns:
            Dictionary of conservative settings
        """
        if not self.cuda_available:
            return {
                'whisper_batch_size': 1,
                'llm_batch_size': 1,
                'video_batch_size': 1,
                'chunk_length_ms': 15000  # Smaller chunks
            }
        
        # Get current memory pressure
        try:
            allocated = torch.cuda.memory_allocated() / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            usage_percent = (allocated / total) * 100
            
            if usage_percent > 80:
                # Very conservative settings
                return {
                    'whisper_batch_size': 2,
                    'llm_batch_size': 1,
                    'video_batch_size': 1,
                    'chunk_length_ms': 15000
                }
            elif usage_percent > 60:
                # Moderately conservative
                return {
                    'whisper_batch_size': 4,
                    'llm_batch_size': 2,
                    'video_batch_size': 1,
                    'chunk_length_ms': 20000
                }
            else:
                # Standard conservative settings
                return {
                    'whisper_batch_size': 8,
                    'llm_batch_size': 4,
                    'video_batch_size': 2,
                    'chunk_length_ms': 30000
                }
                
        except Exception as e:
            logger.error(f"Conservative settings calculation failed: {e}")
            return {
                'whisper_batch_size': 4,
                'llm_batch_size': 2,
                'video_batch_size': 1,
                'chunk_length_ms': 20000
            }
    
    def monitor_batch_performance(self, batch_size: int, success: bool, memory_peak_gb: float):
        """
        Monitor batch performance for future optimization.
        
        Args:
            batch_size: Size of processed batch
            success: Whether batch completed successfully
            memory_peak_gb: Peak memory usage during batch
        """
        if batch_size not in self.batch_performance:
            self.batch_performance[batch_size] = {
                'successes': 0,
                'failures': 0,
                'avg_memory': 0,
                'max_memory': 0
            }
        
        stats = self.batch_performance[batch_size]
        
        if success:
            stats['successes'] += 1
        else:
            stats['failures'] += 1
        
        # Update memory statistics
        total_runs = stats['successes'] + stats['failures']
        stats['avg_memory'] = (stats['avg_memory'] * (total_runs - 1) + memory_peak_gb) / total_runs
        stats['max_memory'] = max(stats['max_memory'], memory_peak_gb)
        
        # Log performance insights
        success_rate = stats['successes'] / total_runs * 100
        logger.info(f"Batch size {batch_size} performance: {success_rate:.1f}% success, "
                   f"avg memory: {stats['avg_memory']:.1f}GB, max: {stats['max_memory']:.1f}GB")
    
    def get_recommended_batch_size(self) -> int:
        """
        Get recommended batch size based on historical performance.
        
        Returns:
            Recommended batch size
        """
        if not self.batch_performance:
            return config.DEFAULT_BATCH_SIZE
        
        # Find batch size with best success rate and reasonable memory usage
        best_batch_size = config.DEFAULT_BATCH_SIZE
        best_score = 0
        
        for batch_size, stats in self.batch_performance.items():
            total_runs = stats['successes'] + stats['failures']
            if total_runs < 2:  # Need at least 2 runs for reliable data
                continue
            
            success_rate = stats['successes'] / total_runs
            memory_efficiency = 1.0 / (1.0 + stats['avg_memory'])  # Prefer lower memory
            
            # Combined score: success rate + memory efficiency
            score = success_rate * 0.7 + memory_efficiency * 0.3
            
            if score > best_score:
                best_score = score
                best_batch_size = batch_size
        
        logger.info(f"Recommended batch size based on history: {best_batch_size}")
        return best_batch_size