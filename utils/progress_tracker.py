"""
Progress tracking utilities for monitoring video processing stages.
Provides real-time updates, time estimation, and detailed progress reporting.
"""
import time
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
import json
from pathlib import Path
import threading
import queue
import logging
from dataclasses import dataclass, asdict

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingStage:
    """Represents a processing stage with timing and progress information."""
    name: str
    display_name: str
    weight: float  # Relative weight for overall progress calculation
    estimated_duration_seconds: float = 0
    actual_duration_seconds: float = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    progress_percent: float = 0.0
    status: str = 'pending'  # pending, active, completed, failed
    error_message: Optional[str] = None
    substages: List[str] = None

@dataclass
class VideoProcessingProgress:
    """Represents progress for a single video."""
    video_id: int
    file_path: str
    filename: str
    total_progress_percent: float = 0.0
    current_stage: Optional[str] = None
    stages: Dict[str, ProcessingStage] = None
    start_time: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    processing_rate: float = 0.0  # MB/s or similar
    status: str = 'queued'

class ProgressTracker:
    """Tracks and manages progress for video processing operations."""
    
    def __init__(self):
        """Initialize progress tracker."""
        
        # Define processing stages with weights and estimates
        self.stage_definitions = {
            'scene_detection': ProcessingStage(
                name='scene_detection',
                display_name='Scene Detection',
                weight=0.2,
                estimated_duration_seconds=60,  # 1 minute
                substages=['loading_video', 'analyzing_frames', 'detecting_transitions', 'extracting_frames']
            ),
            'transcription': ProcessingStage(
                name='transcription',
                display_name='Audio Transcription',
                weight=0.3,
                estimated_duration_seconds=180,  # 3 minutes
                substages=['extracting_audio', 'loading_whisper', 'transcribing', 'processing_segments']
            ),
            'scene_analysis': ProcessingStage(
                name='scene_analysis',
                display_name='Scene Importance Analysis',
                weight=0.1,
                estimated_duration_seconds=30,  # 30 seconds
                substages=['calculating_importance', 'analyzing_content', 'scoring_scenes']
            ),
            'narrative_analysis': ProcessingStage(
                name='narrative_analysis',
                display_name='Narrative Analysis',
                weight=0.2,
                estimated_duration_seconds=120,  # 2 minutes
                substages=['loading_llm', 'analyzing_structure', 'identifying_moments', 'generating_insights']
            ),
            'summary_creation': ProcessingStage(
                name='summary_creation',
                display_name='Summary Creation',
                weight=0.15,
                estimated_duration_seconds=90,  # 1.5 minutes
                substages=['selecting_scenes', 'rendering_video', 'creating_bookmarks']
            ),
            'validation': ProcessingStage(
                name='validation',
                display_name='Quality Validation',
                weight=0.05,
                estimated_duration_seconds=15,  # 15 seconds
                substages=['calculating_metrics', 'validating_output']
            )
        }
        
        # Active progress tracking
        self.active_videos: Dict[int, VideoProcessingProgress] = {}
        self.completed_videos: List[Dict] = []
        self.progress_callbacks: List[Callable] = []
        
        # Statistics
        self.processing_stats = {
            'total_started': 0,
            'total_completed': 0,
            'total_failed': 0,
            'average_processing_time': 0.0,
            'current_throughput': 0.0
        }
        
        # Progress update queue for thread safety
        self.update_queue = queue.Queue()
        self.update_thread = None
        self.running = False
        
        # Historical data for estimation improvement
        self.historical_timings: Dict[str, List[float]] = {
            stage: [] for stage in self.stage_definitions.keys()
        }
        
        self.start_update_thread()
    
    def start_update_thread(self):
        """Start the progress update processing thread."""
        if self.update_thread and self.update_thread.is_alive():
            return
        
        self.running = True
        self.update_thread = threading.Thread(target=self._process_updates, daemon=True)
        self.update_thread.start()
        logger.info("Progress tracker update thread started")
    
    def stop_update_thread(self):
        """Stop the progress update processing thread."""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)
        logger.info("Progress tracker update thread stopped")
    
    def _process_updates(self):
        """Process progress updates from the queue."""
        while self.running:
            try:
                update = self.update_queue.get(timeout=1)
                self._apply_update(update)
                self.update_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing progress update: {e}")
    
    def start_video_processing(self, video_id: int, file_path: str) -> VideoProcessingProgress:
        """
        Start tracking progress for a video.
        
        Args:
            video_id: Database ID of the video
            file_path: Path to the video file
            
        Returns:
            VideoProcessingProgress object
        """
        filename = Path(file_path).name
        
        # Initialize stages
        stages = {}
        for stage_name, stage_def in self.stage_definitions.items():
            stages[stage_name] = ProcessingStage(
                name=stage_def.name,
                display_name=stage_def.display_name,
                weight=stage_def.weight,
                estimated_duration_seconds=self._get_estimated_duration(stage_name, file_path),
                substages=stage_def.substages.copy() if stage_def.substages else []
            )
        
        progress = VideoProcessingProgress(
            video_id=video_id,
            file_path=file_path,
            filename=filename,
            stages=stages,
            start_time=datetime.now(),
            status='processing'
        )
        
        self.active_videos[video_id] = progress
        self.processing_stats['total_started'] += 1
        
        # Update estimated completion
        self._update_estimated_completion(video_id)
        
        logger.info(f"Started tracking progress for video {video_id}: {filename}")
        self._notify_callbacks()
        
        return progress
    
    def update_stage_progress(self, video_id: int, stage_name: str, 
                            progress_percent: float, substage: str = None):
        """
        Update progress for a specific stage.
        
        Args:
            video_id: Video ID
            stage_name: Name of the processing stage
            progress_percent: Progress percentage (0-100)
            substage: Optional substage name
        """
        update = {
            'type': 'stage_progress',
            'video_id': video_id,
            'stage_name': stage_name,
            'progress_percent': progress_percent,
            'substage': substage,
            'timestamp': datetime.now()
        }
        
        self.update_queue.put(update)
    
    def start_stage(self, video_id: int, stage_name: str):
        """Mark a stage as started."""
        update = {
            'type': 'start_stage',
            'video_id': video_id,
            'stage_name': stage_name,
            'timestamp': datetime.now()
        }
        
        self.update_queue.put(update)
    
    def complete_stage(self, video_id: int, stage_name: str, success: bool = True, 
                      error_message: str = None):
        """Mark a stage as completed."""
        update = {
            'type': 'complete_stage',
            'video_id': video_id,
            'stage_name': stage_name,
            'success': success,
            'error_message': error_message,
            'timestamp': datetime.now()
        }
        
        self.update_queue.put(update)
    
    def complete_video_processing(self, video_id: int, success: bool = True, 
                                error_message: str = None):
        """Mark video processing as completed."""
        update = {
            'type': 'complete_video',
            'video_id': video_id,
            'success': success,
            'error_message': error_message,
            'timestamp': datetime.now()
        }
        
        self.update_queue.put(update)
    
    def _apply_update(self, update: Dict):
        """Apply a progress update to the tracking data."""
        video_id = update['video_id']
        
        if video_id not in self.active_videos:
            logger.warning(f"Received update for unknown video {video_id}")
            return
        
        progress = self.active_videos[video_id]
        
        if update['type'] == 'stage_progress':
            self._update_stage_progress(progress, update)
        elif update['type'] == 'start_stage':
            self._start_stage(progress, update)
        elif update['type'] == 'complete_stage':
            self._complete_stage(progress, update)
        elif update['type'] == 'complete_video':
            self._complete_video(progress, update)
        
        # Update overall progress
        self._calculate_overall_progress(progress)
        self._update_estimated_completion(video_id)
        
        # Notify callbacks
        self._notify_callbacks()
    
    def _update_stage_progress(self, progress: VideoProcessingProgress, update: Dict):
        """Update progress for a specific stage."""
        stage_name = update['stage_name']
        
        if stage_name in progress.stages:
            stage = progress.stages[stage_name]
            stage.progress_percent = min(100.0, max(0.0, update['progress_percent']))
            
            if stage.status == 'pending':
                stage.status = 'active'
                stage.start_time = update['timestamp']
            
            progress.current_stage = stage_name
            
            # Update substage if provided
            if update.get('substage'):
                # Could track substage progress here if needed
                pass
    
    def _start_stage(self, progress: VideoProcessingProgress, update: Dict):
        """Mark a stage as started."""
        stage_name = update['stage_name']
        
        if stage_name in progress.stages:
            stage = progress.stages[stage_name]
            stage.status = 'active'
            stage.start_time = update['timestamp']
            stage.progress_percent = 0.0
            
            progress.current_stage = stage_name
            
            logger.debug(f"Started stage {stage_name} for video {progress.video_id}")
    
    def _complete_stage(self, progress: VideoProcessingProgress, update: Dict):
        """Mark a stage as completed."""
        stage_name = update['stage_name']
        
        if stage_name in progress.stages:
            stage = progress.stages[stage_name]
            stage.end_time = update['timestamp']
            stage.progress_percent = 100.0
            
            if update['success']:
                stage.status = 'completed'
                
                # Calculate actual duration and update historical data
                if stage.start_time:
                    actual_duration = (stage.end_time - stage.start_time).total_seconds()
                    stage.actual_duration_seconds = actual_duration
                    self.historical_timings[stage_name].append(actual_duration)
                    
                    # Keep only recent timings for better estimation
                    if len(self.historical_timings[stage_name]) > 50:
                        self.historical_timings[stage_name] = self.historical_timings[stage_name][-50:]
            else:
                stage.status = 'failed'
                stage.error_message = update.get('error_message')
            
            logger.debug(f"Completed stage {stage_name} for video {progress.video_id} "
                        f"(success: {update['success']})")
    
    def _complete_video(self, progress: VideoProcessingProgress, update: Dict):
        """Mark video processing as completed."""
        if update['success']:
            progress.status = 'completed'
            progress.total_progress_percent = 100.0
            self.processing_stats['total_completed'] += 1
        else:
            progress.status = 'failed'
            self.processing_stats['total_failed'] += 1
        
        # Calculate total processing time
        if progress.start_time:
            total_time = (update['timestamp'] - progress.start_time).total_seconds()
            
            # Update average processing time
            total_processed = self.processing_stats['total_completed'] + self.processing_stats['total_failed']
            if total_processed > 0:
                current_avg = self.processing_stats['average_processing_time']
                self.processing_stats['average_processing_time'] = (
                    (current_avg * (total_processed - 1) + total_time) / total_processed
                )
        
        # Move to completed list
        self.completed_videos.append({
            'video_id': progress.video_id,
            'filename': progress.filename,
            'status': progress.status,
            'total_time_seconds': total_time if progress.start_time else 0,
            'completed_at': update['timestamp'].isoformat()
        })
        
        # Remove from active tracking
        del self.active_videos[progress.video_id]
        
        logger.info(f"Completed processing for video {progress.video_id}: {progress.filename} "
                   f"(success: {update['success']})")
    
    def _calculate_overall_progress(self, progress: VideoProcessingProgress):
        """Calculate overall progress percentage based on stage weights."""
        total_progress = 0.0
        total_weight = 0.0
        
        for stage in progress.stages.values():
            stage_progress = stage.progress_percent / 100.0
            total_progress += stage_progress * stage.weight
            total_weight += stage.weight
        
        progress.total_progress_percent = (total_progress / total_weight) * 100.0 if total_weight > 0 else 0.0
    
    def _update_estimated_completion(self, video_id: int):
        """Update estimated completion time for a video."""
        if video_id not in self.active_videos:
            return
        
        progress = self.active_videos[video_id]
        
        if not progress.start_time:
            return
        
        # Calculate time elapsed and remaining work
        elapsed = (datetime.now() - progress.start_time).total_seconds()
        remaining_weight = 0.0
        estimated_remaining_time = 0.0
        
        for stage in progress.stages.values():
            if stage.status == 'completed':
                continue
            elif stage.status == 'active':
                # Partially completed stage
                remaining_progress = (100.0 - stage.progress_percent) / 100.0
                remaining_weight += stage.weight * remaining_progress
                
                # Estimate remaining time for this stage
                if stage.start_time and stage.progress_percent > 0:
                    stage_elapsed = (datetime.now() - stage.start_time).total_seconds()
                    estimated_stage_total = stage_elapsed / (stage.progress_percent / 100.0)
                    estimated_remaining_time += estimated_stage_total - stage_elapsed
                else:
                    estimated_remaining_time += self._get_estimated_duration(stage.name, progress.file_path)
            else:
                # Pending stage
                remaining_weight += stage.weight
                estimated_remaining_time += self._get_estimated_duration(stage.name, progress.file_path)
        
        if estimated_remaining_time > 0:
            progress.estimated_completion = datetime.now() + timedelta(seconds=estimated_remaining_time)
    
    def _get_estimated_duration(self, stage_name: str, file_path: str) -> float:
        """Get estimated duration for a stage based on historical data and file characteristics."""
        
        # Use historical data if available
        if stage_name in self.historical_timings and self.historical_timings[stage_name]:
            timings = self.historical_timings[stage_name]
            # Use median of recent timings
            timings.sort()
            median_time = timings[len(timings) // 2]
            return median_time
        
        # Fall back to default estimates, adjusted by file size if possible
        base_estimate = self.stage_definitions[stage_name].estimated_duration_seconds
        
        try:
            file_size_gb = Path(file_path).stat().st_size / (1024**3)
            
            # Scale estimate based on file size (rough approximation)
            if file_size_gb > 2:  # Large files take longer
                size_multiplier = min(2.0, 1.0 + (file_size_gb - 2) * 0.2)
                base_estimate *= size_multiplier
            
        except Exception:
            pass  # Use base estimate if file size check fails
        
        return base_estimate
    
    def _notify_callbacks(self):
        """Notify all registered progress callbacks."""
        for callback in self.progress_callbacks:
            try:
                callback(self.get_progress_summary())
            except Exception as e:
                logger.error(f"Progress callback failed: {e}")
    
    def add_progress_callback(self, callback: Callable):
        """Add a callback function to be notified of progress updates."""
        self.progress_callbacks.append(callback)
    
    def remove_progress_callback(self, callback: Callable):
        """Remove a progress callback."""
        if callback in self.progress_callbacks:
            self.progress_callbacks.remove(callback)
    
    def get_progress_summary(self) -> Dict:
        """Get a summary of all current progress."""
        active_videos = []
        
        for video_id, progress in self.active_videos.items():
            active_videos.append({
                'video_id': video_id,
                'filename': progress.filename,
                'total_progress': progress.total_progress_percent,
                'current_stage': progress.current_stage,
                'status': progress.status,
                'estimated_completion': progress.estimated_completion.isoformat() if progress.estimated_completion else None,
                'stages': {
                    name: {
                        'display_name': stage.display_name,
                        'progress': stage.progress_percent,
                        'status': stage.status
                    }
                    for name, stage in progress.stages.items()
                }
            })
        
        return {
            'active_videos': active_videos,
            'statistics': self.processing_stats,
            'completed_videos': self.completed_videos[-10:],  # Last 10 completed
            'queue_size': len(self.active_videos)
        }
    
    def get_video_progress(self, video_id: int) -> Optional[Dict]:
        """Get detailed progress for a specific video."""
        if video_id not in self.active_videos:
            return None
        
        progress = self.active_videos[video_id]
        
        return {
            'video_id': video_id,
            'filename': progress.filename,
            'file_path': progress.file_path,
            'total_progress': progress.total_progress_percent,
            'current_stage': progress.current_stage,
            'status': progress.status,
            'start_time': progress.start_time.isoformat() if progress.start_time else None,
            'estimated_completion': progress.estimated_completion.isoformat() if progress.estimated_completion else None,
            'stages': {
                name: asdict(stage) for name, stage in progress.stages.items()
            }
        }
    
    def export_progress_data(self, output_path: str):
        """Export progress tracking data for analysis."""
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'statistics': self.processing_stats,
            'historical_timings': self.historical_timings,
            'stage_definitions': {
                name: asdict(stage) for name, stage in self.stage_definitions.items()
            },
            'completed_videos': self.completed_videos,
            'active_videos': [
                asdict(progress) for progress in self.active_videos.values()
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Progress data exported to: {output_path}")
    
    def cleanup(self):
        """Cleanup progress tracker resources."""
        logger.info("Cleaning up progress tracker...")
        self.stop_update_thread()
        self.active_videos.clear()
        self.progress_callbacks.clear()
