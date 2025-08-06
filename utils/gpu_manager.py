"""
GPU management utilities for monitoring and optimizing CUDA usage.
Handles GPU memory management, temperature monitoring, and resource allocation.
"""
import torch
import psutil
import subprocess
import json
from typing import Dict, Optional, List
import logging
import time
from pathlib import Path

from config_detector import CONFIG as config, ENVIRONMENT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPUManager:
    """Manages GPU resources and monitoring for video processing."""
    
    def __init__(self):
        """Initialize GPU manager with environment detection."""
        self.cuda_available = torch.cuda.is_available()
        self.device_count = torch.cuda.device_count() if self.cuda_available else 0
        
        if ENVIRONMENT == "rtx_3060_local":
            # RTX 3060 optimization
            self.primary_device = config.CUDA_DEVICE
            self.max_memory_gb = config.MAX_GPU_MEMORY_GB
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.cuda.set_per_process_memory_fraction(config.GPU_MEMORY_FRACTION)
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU Manager initialized - RTX 3060: {gpu_name}")
        elif self.cuda_available:
            # Generic CUDA
            self.primary_device = "cuda"
            self.max_memory_gb = 8  # Conservative default
            logger.info(f"GPU Manager initialized - CUDA: {torch.cuda.get_device_name(0)}")
        else:
            # CPU fallback
            self.primary_device = "cpu"
            self.max_memory_gb = 0
            logger.info("GPU Manager initialized - CPU mode")
        
        # Memory management
        self.memory_reserved = 0
        self.memory_checkpoints = []
    
    def get_gpu_info(self) -> Dict:
        """
        Get comprehensive GPU information.
        
        Returns:
            Dictionary containing GPU status and specifications
        """
        gpu_info = {
            'cuda_available': True,
            'device_count': self.device_count,
            'primary_device': self.primary_device,
            'gpu_architecture': 'RTX 3060 TUF'
        }
        
        try:
            # Get primary GPU information
            device_props = torch.cuda.get_device_properties(0)
            
            gpu_info.update({
                'gpu_name': device_props.name,
                'compute_capability': f"{device_props.major}.{device_props.minor}",
                'total_memory_gb': device_props.total_memory / (1024**3),
                'multiprocessor_count': device_props.multi_processor_count
            })
            
            # Current memory usage
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                memory_reserved = torch.cuda.memory_reserved() / (1024**3)
                max_memory_allocated = torch.cuda.max_memory_allocated() / (1024**3)
                
                gpu_info.update({
                    'memory_allocated_gb': memory_allocated,
                    'memory_reserved_gb': memory_reserved,
                    'max_memory_allocated_gb': max_memory_allocated,
                    'memory_usage_percent': (memory_allocated / gpu_info['total_memory_gb']) * 100
                })
            
            # CUDA version information
            gpu_info['cuda_version'] = torch.version.cuda
            gpu_info['cudnn_version'] = torch.backends.cudnn.version()
            
            # Try to get additional info via nvidia-smi
            nvidia_info = self._get_nvidia_smi_info()
            if nvidia_info:
                gpu_info.update(nvidia_info)
            
        except Exception as e:
            logger.error(f"Failed to get GPU info: {e}")
            gpu_info['error'] = str(e)
        
        return gpu_info
    
    def _get_nvidia_smi_info(self) -> Optional[Dict]:
        """Get additional GPU information using nvidia-smi."""
        try:
            # Run nvidia-smi with JSON output
            result = subprocess.run([
                'nvidia-smi',
                '--query-gpu=temperature.gpu,utilization.gpu,utilization.memory,power.draw',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                values = result.stdout.strip().split(', ')
                return {
                    'temperature': int(values[0]),
                    'utilization_gpu': int(values[1]),
                    'utilization_memory': int(values[2]),
                    'power_draw': float(values[3])
                }
        
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError, ValueError):
            logger.debug("nvidia-smi not available or failed")
        
        return None
    
    def check_memory_availability(self, required_gb: float) -> bool:
        """
        Check if sufficient GPU memory is available.
        
        Args:
            required_gb: Required memory in GB
            
        Returns:
            True if memory is available
        """
        if not self.cuda_available:
            return False
        
        try:
            # Get current memory usage
            allocated = torch.cuda.memory_allocated() / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            available = total - allocated
            
            # Check against our limit and required memory
            max_usable = min(self.max_memory_gb, total * 0.9)  # Leave 10% buffer
            can_allocate = (allocated + required_gb) <= max_usable
            
            logger.debug(f"Memory check: {allocated:.1f}GB used, {required_gb:.1f}GB required, "
                        f"{available:.1f}GB available, {max_usable:.1f}GB max usable")
            
            return can_allocate and available >= required_gb
            
        except Exception as e:
            logger.error(f"Memory availability check failed: {e}")
            return False
    
    def reserve_memory(self, amount_gb: float) -> bool:
        """
        Reserve GPU memory for processing.
        
        Args:
            amount_gb: Amount of memory to reserve in GB
            
        Returns:
            True if memory was successfully reserved
        """
        if not self.cuda_available:
            return False
        
        try:
            if self.check_memory_availability(amount_gb):
                # Create a dummy tensor to reserve memory
                dummy_size = int(amount_gb * 1024**3 / 4)  # Float32 = 4 bytes
                dummy_tensor = torch.empty(dummy_size, device=self.primary_device)
                
                self.memory_reserved += amount_gb
                self.memory_checkpoints.append({
                    'tensor': dummy_tensor,
                    'size_gb': amount_gb,
                    'timestamp': time.time()
                })
                
                logger.info(f"Reserved {amount_gb:.1f}GB GPU memory")
                return True
            else:
                logger.warning(f"Cannot reserve {amount_gb:.1f}GB - insufficient memory")
                return False
                
        except Exception as e:
            logger.error(f"Memory reservation failed: {e}")
            return False
    
    def release_memory(self, amount_gb: Optional[float] = None):
        """
        Release reserved GPU memory.
        
        Args:
            amount_gb: Amount to release, or None to release all
        """
        if not self.cuda_available or not self.memory_checkpoints:
            return
        
        try:
            if amount_gb is None:
                # Release all
                for checkpoint in self.memory_checkpoints:
                    del checkpoint['tensor']
                
                released = self.memory_reserved
                self.memory_checkpoints.clear()
                self.memory_reserved = 0
                
                logger.info(f"Released all reserved memory: {released:.1f}GB")
            else:
                # Release specific amount
                released = 0
                remaining_checkpoints = []
                
                for checkpoint in self.memory_checkpoints:
                    if released < amount_gb:
                        del checkpoint['tensor']
                        released += checkpoint['size_gb']
                        self.memory_reserved -= checkpoint['size_gb']
                    else:
                        remaining_checkpoints.append(checkpoint)
                
                self.memory_checkpoints = remaining_checkpoints
                logger.info(f"Released {released:.1f}GB GPU memory")
            
            # Force garbage collection
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Memory release failed: {e}")
    
    def optimize_memory_usage(self, aggressive: bool = False):
        """
        Optimize GPU memory usage by clearing caches and releasing unused memory.
        
        Args:
            aggressive: If True, performs more aggressive cleanup
        """
        if not self.cuda_available:
            return
        
        try:
            # Get memory before cleanup
            memory_before = torch.cuda.memory_allocated() / (1024**3)
            
            # Standard cleanup
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            if aggressive:
                # Force garbage collection
                import gc
                gc.collect()
                
                # Clear all cached memory
                torch.cuda.empty_cache()
                
                # Reset peak memory stats
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.reset_accumulated_memory_stats()
                
                # Additional synchronization
                torch.cuda.synchronize()
            
            memory_after = torch.cuda.memory_allocated() / (1024**3)
            freed = memory_before - memory_after
            
            logger.info(f"GPU memory optimization completed - Freed {freed:.2f}GB (aggressive={aggressive})")
            
        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")
    
    def check_memory_pressure(self) -> Dict:
        """
        Check current memory pressure and recommend actions.
        
        Returns:
            Dict with memory status and recommendations
        """
        if not self.cuda_available:
            return {'status': 'no_gpu', 'action': 'continue'}
        
        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            
            usage_percent = (allocated / total_memory) * 100
            pressure_level = 'low'
            action = 'continue'
            
            if usage_percent > 90:
                pressure_level = 'critical'
                action = 'emergency_cleanup'
            elif usage_percent > 80:
                pressure_level = 'high'
                action = 'aggressive_cleanup'
            elif usage_percent > 70:
                pressure_level = 'medium' 
                action = 'standard_cleanup'
            
            return {
                'status': 'ok',
                'total_gb': total_memory,
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'usage_percent': usage_percent,
                'pressure_level': pressure_level,
                'action': action,
                'available_gb': total_memory - allocated
            }
            
        except Exception as e:
            logger.error(f"Memory pressure check failed: {e}")
            return {'status': 'error', 'action': 'continue', 'error': str(e)}
    
    def emergency_memory_cleanup(self):
        """Emergency memory cleanup when running out of VRAM."""
        if not self.cuda_available:
            return
            
        try:
            logger.warning("Emergency memory cleanup initiated")
            
            # Release all reserved memory
            self.release_memory()
            
            # Aggressive optimization
            self.optimize_memory_usage(aggressive=True)
            
            # Force Python garbage collection
            import gc
            gc.collect()
            
            # Multiple cache clears
            for _ in range(3):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logger.info("Emergency cleanup completed")
            
        except Exception as e:
            logger.error(f"Emergency cleanup failed: {e}")
    
    def get_memory_stats(self) -> Dict:
        """Get detailed memory statistics."""
        if not self.cuda_available:
            return {'status': 'CUDA not available'}
        
        try:
            stats = {
                'allocated_gb': torch.cuda.memory_allocated() / (1024**3),
                'reserved_gb': torch.cuda.memory_reserved() / (1024**3),
                'max_allocated_gb': torch.cuda.max_memory_allocated() / (1024**3),
                'max_reserved_gb': torch.cuda.max_memory_reserved() / (1024**3),
                'manually_reserved_gb': self.memory_reserved,
                'cache_info': torch.cuda.memory_stats()
            }
            
            # Calculate percentages
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            stats['total_memory_gb'] = total_memory
            stats['usage_percent'] = (stats['allocated_gb'] / total_memory) * 100
            stats['available_gb'] = total_memory - stats['allocated_gb']
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {'error': str(e)}
    
    def monitor_temperature(self) -> Optional[Dict]:
        """Monitor GPU temperature and thermal state."""
        nvidia_info = self._get_nvidia_smi_info()
        
        if nvidia_info and 'temperature' in nvidia_info:
            temp = nvidia_info['temperature']
            
            # Determine thermal state
            if temp < 60:
                thermal_state = 'optimal'
            elif temp < 75:
                thermal_state = 'warm'
            elif temp < 85:
                thermal_state = 'hot'
            else:
                thermal_state = 'critical'
            
            return {
                'temperature_c': temp,
                'thermal_state': thermal_state,
                'power_draw_w': nvidia_info.get('power_draw'),
                'utilization_percent': nvidia_info.get('utilization_gpu')
            }
        
        return None
    
    def suggest_batch_size(self, model_memory_gb: float, 
                          video_memory_per_item_gb: float) -> int:
        """
        Suggest optimal batch size based on available memory.
        
        Args:
            model_memory_gb: Memory required for model
            video_memory_per_item_gb: Memory per video item
            
        Returns:
            Suggested batch size
        """
        if not self.cuda_available:
            return 1
        
        try:
            # Get available memory
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            allocated = torch.cuda.memory_allocated() / (1024**3)
            available = total_memory - allocated
            
            # Reserve memory for model and system
            usable_memory = min(available * 0.8, self.max_memory_gb)  # Use 80% of available
            memory_for_batch = usable_memory - model_memory_gb
            
            if memory_for_batch <= 0:
                return 1
            
            # Calculate batch size
            suggested_batch_size = int(memory_for_batch / video_memory_per_item_gb)
            suggested_batch_size = max(1, min(suggested_batch_size, config.MAX_BATCH_SIZE))
            
            logger.info(f"Suggested batch size: {suggested_batch_size} "
                       f"(available: {available:.1f}GB, usable: {usable_memory:.1f}GB)")
            
            return suggested_batch_size
            
        except Exception as e:
            logger.error(f"Batch size suggestion failed: {e}")
            return 1
    
    def create_memory_checkpoint(self, name: str) -> Dict:
        """
        Create a memory usage checkpoint for monitoring.
        
        Args:
            name: Name for the checkpoint
            
        Returns:
            Checkpoint data
        """
        checkpoint = {
            'name': name,
            'timestamp': time.time(),
            'memory_stats': self.get_memory_stats(),
            'system_memory': psutil.virtual_memory().percent
        }
        
        # Add thermal information if available
        thermal_info = self.monitor_temperature()
        if thermal_info:
            checkpoint['thermal_info'] = thermal_info
        
        logger.debug(f"Memory checkpoint '{name}': {checkpoint['memory_stats']['allocated_gb']:.1f}GB")
        
        return checkpoint
    
    def get_system_compatibility(self) -> Dict:
        """Check system compatibility for GPU processing."""
        compatibility = {
            'cuda_available': self.cuda_available,
            'recommended_setup': True,
            'warnings': [],
            'recommendations': []
        }
        
        if not self.cuda_available:
            compatibility['recommended_setup'] = False
            compatibility['warnings'].append("CUDA not available - processing will be CPU-only")
            compatibility['recommendations'].append("Install CUDA-compatible PyTorch")
            return compatibility
        
        # Check GPU memory
        gpu_info = self.get_gpu_info()
        total_memory = gpu_info.get('total_memory_gb', 0)
        
        if total_memory < 8:
            compatibility['warnings'].append(f"GPU has only {total_memory:.1f}GB memory (recommended: 8GB+)")
            compatibility['recommendations'].append("Consider reducing batch size or model size")
        
        # Check CUDA version
        cuda_version = gpu_info.get('cuda_version')
        if cuda_version and float(cuda_version) < 11.0:
            compatibility['warnings'].append(f"CUDA version {cuda_version} is older (recommended: 11.0+)")
            compatibility['recommendations'].append("Update CUDA drivers")
        
        # Check system memory
        system_memory_gb = psutil.virtual_memory().total / (1024**3)
        if system_memory_gb < 16:
            compatibility['warnings'].append(f"System has only {system_memory_gb:.1f}GB RAM (recommended: 16GB+)")
            compatibility['recommendations'].append("Close other applications during processing")
        
        # Check thermal state
        thermal_info = self.monitor_temperature()
        if thermal_info and thermal_info.get('thermal_state') in ['hot', 'critical']:
            compatibility['warnings'].append(f"GPU temperature is {thermal_info['thermal_state']}")
            compatibility['recommendations'].append("Improve GPU cooling or reduce processing intensity")
        
        if compatibility['warnings']:
            compatibility['recommended_setup'] = False
        
        return compatibility
    
    def cleanup(self):
        """Cleanup GPU resources and release all reserved memory."""
        logger.info("Cleaning up GPU resources...")
        
        try:
            # Release all reserved memory
            self.release_memory()
            
            # Clear PyTorch cache
            if self.cuda_available:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logger.info("GPU cleanup completed")
            
        except Exception as e:
            logger.error(f"GPU cleanup failed: {e}")
