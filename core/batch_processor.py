"""
Batch processing module for handling multiple videos with GPU memory management.
Implements queue management, progress tracking, and error recovery.
"""
import threading
import queue
import time
from pathlib import Path
from typing import List, Dict, Optional, Callable
import logging
from datetime import datetime, timedelta
import psutil
import torch

from .database import VideoDatabase
from .video_discovery import VideoDiscovery
from .scene_detector import SceneDetector
from .transcription import OfflineTranscriber
from .narrative_analyzer import NarrativeAnalyzer
from .summarizer import VideoSummarizer
from .validation import ValidationMetrics
from .vlc_bookmarks import VLCBookmarkGenerator
from .advanced_scene_analysis import AdvancedSceneAnalyzer
from .audio_analysis import AudioAnalyzer
from utils.gpu_manager import GPUManager

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BatchProcessor:
    """Manages batch processing of multiple videos with resource optimization."""
    
    def __init__(self):
        """Initialize batch processor with all required components."""
        self.db = VideoDatabase()
        self.discovery = VideoDiscovery(db=self.db)
        self.scene_detector = SceneDetector()
        self.transcriber = OfflineTranscriber()
        self.analyzer = NarrativeAnalyzer()
        self.summarizer = VideoSummarizer()
        self.validator = ValidationMetrics()
        self.bookmark_generator = VLCBookmarkGenerator()
        self.advanced_scene_analyzer = AdvancedSceneAnalyzer()
        self.audio_analyzer = AudioAnalyzer()
        self.gpu_manager = GPUManager()
        
        # Processing queue and state
        self.processing_queue = queue.Queue()
        self.is_processing = False
        self.is_paused = False
        self.current_batch_size = config.DEFAULT_BATCH_SIZE
        self.max_batch_size = config.MAX_BATCH_SIZE
        
        # Progress tracking
        self.progress_callback = None
        self.status_callback = None
        
        # Error tracking
        self.errors = []
        self.processing_stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'start_time': None,
            'estimated_completion': None
        }
    
    def scan_and_queue_videos(self, root_directory: str, 
                             include_subdirs: bool = True) -> Dict:
        """
        Scan directory for videos and add to processing queue.
        
        Args:
            root_directory: Root directory to scan
            include_subdirs: Whether to scan subdirectories
            
        Returns:
            Scan results summary
        """
        logger.info(f"Scanning directory: {root_directory}")
        
        scan_results = {
            'discovered': 0,
            'new': 0,
            'already_processed': 0,
            'queued': 0,
            'errors': []
        }
        
        try:
            # Discover videos
            for metadata in self.discovery.scan_directory(root_directory, include_subdirs):
                scan_results['discovered'] += 1
                
                try:
                    # Add to database
                    video_id = self.db.add_video(metadata['file_path'], metadata)
                    
                    # Check if already processed
                    status = self.db.get_videos_by_status('completed')
                    already_processed = any(
                        v['file_path'] == metadata['file_path'] for v in status
                    )
                    
                    if already_processed:
                        scan_results['already_processed'] += 1
                    else:
                        # Add to processing queue
                        self.processing_queue.put({
                            'video_id': video_id,
                            'file_path': metadata['file_path'],
                            'metadata': metadata
                        })
                        scan_results['new'] += 1
                        scan_results['queued'] += 1
                
                except Exception as e:
                    error_msg = f"Error processing {metadata.get('file_path', 'unknown')}: {e}"
                    logger.error(error_msg)
                    scan_results['errors'].append(error_msg)
            
            logger.info(f"Scan completed: {scan_results}")
            return scan_results
            
        except Exception as e:
            logger.error(f"Directory scan failed: {e}")
            scan_results['errors'].append(str(e))
            return scan_results
    
    def start_batch_processing(self, batch_size: Optional[int] = None, 
                              progress_callback: Optional[Callable] = None,
                              status_callback: Optional[Callable] = None) -> bool:
        """
        Start batch processing of queued videos.
        
        Args:
            batch_size: Number of videos to process simultaneously
            progress_callback: Function to call with progress updates
            status_callback: Function to call with status updates
            
        Returns:
            True if processing started successfully
        """
        if self.is_processing:
            logger.warning("Batch processing already in progress")
            return False
        
        # Load queued videos from database into processing queue
        self._load_queued_videos()
        
        if self.processing_queue.empty():
            logger.warning("No videos in processing queue")
            return False
        
        # Set processing parameters
        self.current_batch_size = batch_size or self.current_batch_size
        self.progress_callback = progress_callback
        self.status_callback = status_callback
        
        # Reset stats
        self.processing_stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'start_time': datetime.now(),
            'estimated_completion': None
        }
        
        # Start processing thread
        self.is_processing = True
        processing_thread = threading.Thread(target=self._process_batch_worker)
        processing_thread.daemon = True
        processing_thread.start()
        
        logger.info(f"Batch processing started with batch size: {self.current_batch_size}")
        return True
    
    def _load_queued_videos(self):
        """Load videos with 'queued' status from database into processing queue."""
        queued_videos = self.db.get_videos_by_status('queued')
        
        for video in queued_videos:
            # Add to processing queue if not already there
            video_item = {
                'video_id': video['id'],
                'file_path': video['file_path'],
                'metadata': video
            }
            self.processing_queue.put(video_item)
            
        logger.info(f"Loaded {len(queued_videos)} queued videos into processing queue")
    
    def pause_batch_processing(self):
        """Pause batch processing (can be resumed)."""
        logger.info("Pausing batch processing...")
        self.is_paused = True
    
    def resume_batch_processing(self):
        """Resume paused batch processing."""
        logger.info("Resuming batch processing...")
        self.is_paused = False
        
    def stop_batch_processing(self):
        """Stop batch processing completely."""
        logger.info("Stopping batch processing...")
        self.is_processing = False
        self.is_paused = False
        
        # Note: We don't clear the queue anymore to allow resuming
        # The queue will be preserved for potential restart
    
    def _process_batch_worker(self):
        """Worker thread for batch processing."""
        try:
            while self.is_processing and not self.processing_queue.empty():
                # Check if paused
                if self.is_paused:
                    logger.info("Processing paused, waiting...")
                    time.sleep(5)  # Check every 5 seconds
                    continue
                
                # Process videos in batches
                batch = self._get_next_batch()
                
                if batch:
                    self._process_video_batch(batch)
                
                # Check GPU memory and system resources
                if not self._check_system_resources():
                    logger.warning("System resources low, pausing processing")
                    time.sleep(60)  # Wait 1 minute before retrying
                
        except Exception as e:
            logger.error(f"Batch processing worker failed: {e}")
            # Ensure app doesn't crash on processing errors
            self.errors.append(f"Batch processing error: {e}")
        finally:
            self.is_processing = False
            self.is_paused = False
            logger.info("Batch processing completed")
    
    def _get_next_batch(self) -> List[Dict]:
        """Get next batch of videos to process."""
        batch = []
        
        for _ in range(self.current_batch_size):
            try:
                if not self.processing_queue.empty():
                    video_item = self.processing_queue.get_nowait()
                    batch.append(video_item)
                else:
                    break
            except queue.Empty:
                break
        
        return batch
    
    def _process_video_batch(self, batch: List[Dict]):
        """Process a batch of videos with intelligent memory management."""
        logger.info(f"Processing batch of {len(batch)} videos")
        
        for idx, video_item in enumerate(batch):
            if not self.is_processing:
                # Put video back in queue if processing was stopped
                self.processing_queue.put(video_item)
                break
            
            # Check memory pressure before processing each video
            memory_status = self.gpu_manager.check_memory_pressure()
            if memory_status['action'] == 'emergency_cleanup':
                logger.warning(f"Critical memory pressure ({memory_status['usage_percent']:.1f}%) - performing emergency cleanup")
                self.gpu_manager.emergency_memory_cleanup()
            elif memory_status['action'] == 'aggressive_cleanup':
                logger.info(f"High memory pressure ({memory_status['usage_percent']:.1f}%) - performing aggressive cleanup")
                self.gpu_manager.optimize_memory_usage(aggressive=True)
            elif memory_status['action'] == 'standard_cleanup':
                logger.info(f"Medium memory pressure ({memory_status['usage_percent']:.1f}%) - performing standard cleanup")
                self.gpu_manager.optimize_memory_usage()
            
            try:
                # Process video with memory monitoring
                self._process_single_video_with_memory_management(video_item)
                self.processing_stats['successful'] += 1
                
                # Cleanup after each video to prevent memory accumulation
                if idx % 2 == 0:  # Every 2 videos
                    self.gpu_manager.optimize_memory_usage()
                
            except torch.cuda.OutOfMemoryError as oom:
                error_msg = f"CUDA out of memory for {video_item['file_path']}: {oom}"
                logger.error(error_msg)
                
                # Attempt recovery
                logger.info("Attempting memory recovery...")
                self.gpu_manager.emergency_memory_cleanup()
                
                # Try processing again with reduced settings
                try:
                    logger.info("Retrying with emergency settings...")
                    self._process_single_video_emergency_mode(video_item)
                    self.processing_stats['successful'] += 1
                except Exception as retry_error:
                    error_msg = f"Video processing failed even in emergency mode: {video_item['file_path']}: {retry_error}"
                    logger.error(error_msg)
                    self.errors.append(error_msg)
                    self.processing_stats['failed'] += 1
                    
                    safe_error_msg = str(retry_error).encode('utf-8', errors='replace').decode('utf-8')
                    self.db.update_processing_status(
                        video_item['video_id'], 'failed', error_message=safe_error_msg
                    )
                
            except Exception as e:
                error_msg = f"Video processing failed: {video_item['file_path']}: {e}"
                logger.error(error_msg)
                self.errors.append(error_msg)
                self.processing_stats['failed'] += 1
                
                # Update database with error (encode error message safely)
                safe_error_msg = str(e).encode('utf-8', errors='replace').decode('utf-8')
                self.db.update_processing_status(
                    video_item['video_id'], 'failed', error_message=safe_error_msg
                )
                
                # Continue processing other videos instead of crashing
                logger.info(f"Continuing with next video after error in {video_item['file_path']}")
            
            self.processing_stats['total_processed'] += 1
            
            # Update progress
            if self.progress_callback:
                self._update_progress()
    
    def _process_single_video_with_memory_management(self, video_item: Dict):
        """Process a single video with enhanced memory management."""
        return self._process_single_video(video_item)
    
    def _process_single_video_emergency_mode(self, video_item: Dict):
        """Process video with minimal memory usage settings."""
        logger.info(f"Processing {video_item['file_path']} in emergency mode")
        
        # Temporarily reduce batch sizes and model settings
        original_whisper_batch_size = getattr(config, 'WHISPER_BATCH_SIZE', 16)
        original_llm_batch_size = getattr(config, 'LLM_BATCH_SIZE', 8)
        
        # Emergency settings - much smaller memory footprint
        config.WHISPER_BATCH_SIZE = 4
        config.LLM_BATCH_SIZE = 2
        
        try:
            # Force memory cleanup before starting
            self.gpu_manager.emergency_memory_cleanup()
            
            # Process with reduced settings
            self._process_single_video(video_item)
            
        finally:
            # Restore original settings
            config.WHISPER_BATCH_SIZE = original_whisper_batch_size
            config.LLM_BATCH_SIZE = original_llm_batch_size
    
    def _process_single_video(self, video_item: Dict):
        """Process a single video through the complete pipeline."""
        video_id = video_item['video_id']
        file_path = video_item['file_path']
        
        logger.info(f"Processing video: {Path(file_path).name}")
        
        # Update status to processing
        self.db.update_processing_status(video_id, 'processing', 0.0, 'starting')
        
        try:
            # Stage 1: Scene Detection
            self._update_status(video_id, 'scene_detection', 10.0)
            scenes = self.scene_detector.detect_scenes(file_path, method="auto")
            self.db.add_scene_data(video_id, scenes)
            
            # Stage 2: Transcription
            self._update_status(video_id, 'transcription', 30.0)
            transcription_data = self.transcriber.transcribe_video(file_path)
            self.db.add_transcription_data(video_id, transcription_data)
            
            # Stage 3: Scene Importance Calculation
            self._update_status(video_id, 'scene_analysis', 50.0)
            scenes = self.scene_detector.calculate_scene_importance(
                scenes, transcription_data
            )
            
            # Stage 4: Narrative Analysis
            self._update_status(video_id, 'narrative_analysis', 65.0)
            narrative_analysis = self.analyzer.analyze_narrative_structure(
                transcription_data, scenes
            )
            
            # Apply narrative importance to scenes
            scenes = self.analyzer.calculate_scene_narrative_importance(
                scenes, narrative_analysis.get('key_moments', [])
            )
            
            # Stage 5: Summary Creation
            self._update_status(video_id, 'summary_creation', 80.0)
            
            # Validate inputs before summary creation
            if not scenes:
                logger.error(f"No scenes detected for {file_path} - cannot create summary")
                raise RuntimeError("Scene detection failed - no scenes available")
            
            logger.info(f"Creating summary with {len(scenes)} scenes for {Path(file_path).name}")
            summary_result = self.summarizer.create_summary(
                Path(file_path), scenes, transcription_data, narrative_analysis
            )
            
            # Stage 6: Validation
            self._update_status(video_id, 'validation', 90.0)
            validation_metrics = self.validator.calculate_f1_score(
                summary_result['selected_scenes'],
                original_duration=summary_result['original_duration']
            )
            
            # Save validation scores
            for metric_name, score in validation_metrics.items():
                if isinstance(score, (int, float)):
                    self.db.add_validation_score(video_id, metric_name, score)
            
            # Save summary data
            self.db.add_summary_data(video_id, summary_result)
            
            # Stage 7: Complete
            self._update_status(video_id, 'completed', 100.0)
            
            logger.info(f"Successfully processed: {Path(file_path).name}")
            
        except Exception as e:
            self.db.update_processing_status(
                video_id, 'failed', error_message=str(e)
            )
            raise
    
    def _update_status(self, video_id: int, stage: str, progress: float):
        """Update processing status and progress."""
        self.db.update_processing_status(video_id, 'processing', progress, stage)
        
        if self.status_callback:
            self.status_callback({
                'video_id': video_id,
                'stage': stage,
                'progress': progress
            })
    
    def _update_progress(self):
        """Update overall progress and estimated completion time."""
        if self.progress_callback:
            # Calculate estimated completion time
            if self.processing_stats['total_processed'] > 0:
                elapsed = datetime.now() - self.processing_stats['start_time']
                rate = self.processing_stats['total_processed'] / elapsed.total_seconds()
                remaining = self.processing_queue.qsize()
                estimated_remaining = timedelta(seconds=remaining / rate) if rate > 0 else None
                
                self.processing_stats['estimated_completion'] = (
                    datetime.now() + estimated_remaining if estimated_remaining else None
                )
            
            self.progress_callback(self.processing_stats.copy())
    
    def _check_system_resources(self) -> bool:
        """Check if system has sufficient resources to continue processing."""
        try:
            # Enhanced GPU memory check using GPU manager
            if torch.cuda.is_available():
                memory_status = self.gpu_manager.check_memory_pressure()
                if memory_status['pressure_level'] == 'critical':
                    logger.warning(f"GPU memory critically high: {memory_status['usage_percent']:.1f}%")
                    # Attempt emergency cleanup
                    self.gpu_manager.emergency_memory_cleanup()
                    # Recheck after cleanup
                    memory_status = self.gpu_manager.check_memory_pressure()
                    if memory_status['pressure_level'] == 'critical':
                        return False
            
            # Check system memory
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 85:
                logger.warning(f"System memory usage high: {memory_percent}%")
                return False
            
            # Check disk space
            try:
                disk_usage = psutil.disk_usage('/').percent
                if disk_usage > 90:
                    logger.warning(f"Disk usage high: {disk_usage}%")
                    return False
            except:
                # Handle cases where disk usage check fails (e.g., Windows)
                pass
            
            return True
            
        except Exception as e:
            logger.error(f"Resource check failed: {e}")
            return True  # Continue processing if check fails
    
    def get_processing_status(self) -> Dict:
        """Get current processing status and statistics."""
        queue_size = self.processing_queue.qsize()
        
        status = {
            'is_processing': self.is_processing,
            'queue_size': queue_size,
            'current_batch_size': self.current_batch_size,
            'stats': self.processing_stats.copy(),
            'errors': self.errors[-10:],  # Last 10 errors
            'resource_usage': self._get_resource_usage()
        }
        
        return status
    
    def _get_resource_usage(self) -> Dict:
        """Get current resource usage."""
        usage = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent
        }
        
        if torch.cuda.is_available():
            usage['gpu_memory_gb'] = torch.cuda.memory_allocated() / (1024**3)
            usage['gpu_memory_percent'] = (
                torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
                if torch.cuda.max_memory_allocated() > 0 else 0
            )
        
        return usage
    
    def clear_queue(self):
        """Clear both the in-memory processing queue and database queue status."""
        # Clear in-memory queue
        while not self.processing_queue.empty():
            try:
                self.processing_queue.get_nowait()
            except queue.Empty:
                break
        
        # Also clear queued status in database
        cleared_count = self.db.clear_all_queued_videos()
        
        logger.info(f"Processing queue cleared - {cleared_count} videos updated in database")
        return cleared_count
    
    def add_video_to_queue_by_id(self, video_id: int) -> bool:
        """Add a video to processing queue by database ID."""
        try:
            video_details = self.db.get_video_details(video_id)
            if not video_details:
                logger.error(f"Video {video_id} not found in database")
                return False
            
            # Set status to queued
            self.db.update_processing_status(video_id, 'queued')
            
            # Add to processing queue
            self.processing_queue.put(video_id)
            
            logger.info(f"Added video {video_id} to processing queue")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add video {video_id} to queue: {e}")
            return False
    
    def add_video_to_queue(self, video_path: str) -> bool:
        """
        Add a specific video to the processing queue.
        
        Args:
            video_path: Path to video file
            
        Returns:
            True if added successfully
        """
        try:
            video_path_obj = Path(video_path)
            
            if not video_path_obj.exists():
                logger.error(f"Video file not found: {video_path_obj}")
                return False
            
            # Extract metadata
            metadata = self.discovery.extract_metadata(video_path_obj)
            if not metadata:
                logger.error(f"Failed to extract metadata: {video_path_obj}")
                return False
            
            # Add to database
            video_id = self.db.add_video(str(video_path_obj), metadata)
            
            # Add to queue
            self.processing_queue.put({
                'video_id': video_id,
                'file_path': str(video_path_obj),
                'metadata': metadata
            })
            
            logger.info(f"Added to queue: {video_path_obj.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add video to queue: {e}")
            return False
    
    def set_batch_size(self, batch_size: int):
        """Set the batch size for processing."""
        if 1 <= batch_size <= self.max_batch_size:
            self.current_batch_size = batch_size
            logger.info(f"Batch size set to: {batch_size}")
        else:
            logger.error(f"Invalid batch size: {batch_size} (must be 1-{self.max_batch_size})")
    
    def get_queue_summary(self) -> Dict:
        """Get summary of videos in the processing queue."""
        # Get actual queued count from database for accurate display
        queued_videos = self.db.get_videos_by_status('queued')
        db_stats = self.db.get_processing_stats()
        
        return {
            'total_in_queue': len(queued_videos),
            'database_stats': db_stats,
            'batch_size': self.current_batch_size,
            'max_batch_size': self.max_batch_size,
            'is_processing': self.is_processing,
            'is_paused': self.is_paused
        }
