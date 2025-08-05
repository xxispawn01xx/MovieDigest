"""
Scene detection module using PySceneDetect and OpenCV.
Combines content-based and adaptive detection for robust scene boundary identification.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector, AdaptiveDetector, ThresholdDetector

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SceneDetector:
    """Advanced scene detection using multiple algorithms."""
    
    def __init__(self):
        """Initialize scene detector with configurable parameters."""
        self.content_threshold = config.SCENE_DETECTION_THRESHOLD
        self.adaptive_threshold = config.ADAPTIVE_THRESHOLD
        self.min_scene_length = config.MIN_SCENE_LENGTH_SECONDS
    
    def detect_scenes(self, video_path: str, method: str = "auto") -> List[Dict]:
        """
        Detect scenes in video using specified method.
        
        Args:
            video_path: Path to video file
            method: Detection method ('content', 'adaptive', 'threshold', 'auto')
            
        Returns:
            List of scene dictionaries with timing and metadata
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        logger.info(f"Starting scene detection for: {video_path.name}")
        logger.info(f"Using method: {method}")
        
        try:
            # Open video and create scene manager
            video = open_video(str(video_path))
            scene_manager = SceneManager()
            
            # Add detectors based on method
            if method == "content":
                scene_manager.add_detector(ContentDetector(threshold=self.content_threshold))
            elif method == "adaptive":
                scene_manager.add_detector(AdaptiveDetector(adaptive_threshold=self.adaptive_threshold))
            elif method == "threshold":
                scene_manager.add_detector(ThresholdDetector(threshold=15.0))
            elif method == "auto":
                # Use multiple detectors for comprehensive detection
                scene_manager.add_detector(ContentDetector(threshold=self.content_threshold))
                scene_manager.add_detector(AdaptiveDetector(adaptive_threshold=self.adaptive_threshold))
            else:
                raise ValueError(f"Unknown detection method: {method}")
            
            # Detect scenes
            scene_manager.detect_scenes(video, show_progress=True)
            scene_list = scene_manager.get_scene_list()
            
            # Process scenes and add metadata
            scenes = self._process_scene_list(scene_list, video_path)
            
            # Filter scenes by minimum length
            scenes = self._filter_scenes_by_length(scenes)
            
            logger.info(f"Detected {len(scenes)} scenes")
            return scenes
            
        except Exception as e:
            logger.error(f"Scene detection failed for {video_path}: {e}")
            raise
    
    def _process_scene_list(self, scene_list: List, video_path: Path) -> List[Dict]:
        """Process PySceneDetect scene list into standardized format."""
        scenes = []
        
        for i, (start_time, end_time) in enumerate(scene_list):
            scene = {
                'scene_number': i + 1,
                'start_time': start_time.get_seconds(),
                'end_time': end_time.get_seconds(),
                'duration': (end_time - start_time).get_seconds(),
                'start_frame': start_time.frame_num,
                'end_frame': end_time.frame_num,
                'frame_count': end_time.frame_num - start_time.frame_num
            }
            
            # Extract representative frame
            try:
                frame_path = self._extract_scene_frame(video_path, scene)
                scene['frame_path'] = frame_path
            except Exception as e:
                logger.warning(f"Failed to extract frame for scene {i+1}: {e}")
                scene['frame_path'] = None
            
            scenes.append(scene)
        
        return scenes
    
    def _extract_scene_frame(self, video_path: Path, scene: Dict) -> Optional[str]:
        """Extract a representative frame from the middle of a scene."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            # Calculate middle frame of scene
            middle_time = (scene['start_time'] + scene['end_time']) / 2
            fps = cap.get(cv2.CAP_PROP_FPS)
            middle_frame = int(middle_time * fps)
            
            # Seek to middle frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
            ret, frame = cap.read()
            
            if ret:
                # Create output directory
                output_dir = config.OUTPUT_DIR / "scene_frames" / video_path.stem
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Save frame
                frame_filename = f"scene_{scene['scene_number']:03d}.jpg"
                frame_path = output_dir / frame_filename
                cv2.imwrite(str(frame_path), frame)
                
                cap.release()
                return str(frame_path)
            
            cap.release()
            return None
            
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            return None
    
    def _filter_scenes_by_length(self, scenes: List[Dict]) -> List[Dict]:
        """Filter out scenes shorter than minimum length."""
        filtered_scenes = []
        
        for scene in scenes:
            if scene['duration'] >= self.min_scene_length:
                filtered_scenes.append(scene)
            else:
                logger.debug(f"Filtered out short scene: {scene['duration']:.2f}s")
        
        logger.info(f"Filtered {len(scenes) - len(filtered_scenes)} short scenes")
        return filtered_scenes
    
    def calculate_scene_importance(self, scenes: List[Dict], 
                                 transcription_data: List[Dict] = None) -> List[Dict]:
        """
        Calculate importance scores for scenes based on multiple factors.
        
        Args:
            scenes: List of scene dictionaries
            transcription_data: Optional transcription data for text analysis
            
        Returns:
            Scenes with added importance scores
        """
        logger.info("Calculating scene importance scores")
        
        for scene in scenes:
            score = 0.0
            
            # Duration-based scoring (longer scenes may be more important)
            duration_score = min(scene['duration'] / 60.0, 1.0)  # Normalize to max 1 minute
            score += duration_score * 0.3
            
            # Position-based scoring (beginning and end scenes often important)
            total_scenes = len(scenes)
            position = scene['scene_number']
            
            if position <= 3 or position >= total_scenes - 2:
                score += 0.4  # Opening/closing scenes bonus
            elif position == total_scenes // 2:
                score += 0.2  # Middle scene bonus
            
            # Transcription-based scoring if available
            if transcription_data:
                text_score = self._calculate_text_importance(scene, transcription_data)
                score += text_score * 0.4
            
            # Visual complexity scoring (placeholder - would need frame analysis)
            visual_score = 0.3  # Default moderate importance
            score += visual_score * 0.2
            
            # Normalize final score to 0-1 range
            scene['importance_score'] = min(score, 1.0)
        
        return scenes
    
    def _calculate_text_importance(self, scene: Dict, transcription_data: List[Dict]) -> float:
        """Calculate importance based on transcription text within scene timeframe."""
        scene_start = scene['start_time']
        scene_end = scene['end_time']
        
        # Find transcription segments within scene
        scene_text = []
        for trans in transcription_data:
            if (trans['start_time'] >= scene_start and trans['end_time'] <= scene_end):
                scene_text.append(trans['text'])
        
        if not scene_text:
            return 0.1  # Low importance if no speech
        
        combined_text = ' '.join(scene_text).lower()
        
        # Simple keyword-based importance scoring
        important_keywords = [
            'important', 'critical', 'key', 'main', 'primary', 'essential',
            'conclusion', 'summary', 'result', 'outcome', 'decision',
            'plot', 'twist', 'reveal', 'discovery', 'climax'
        ]
        
        keyword_count = sum(1 for keyword in important_keywords if keyword in combined_text)
        
        # Text length scoring
        text_length_score = min(len(combined_text) / 500.0, 1.0)
        
        # Combine scores
        importance = (keyword_count * 0.1) + (text_length_score * 0.4)
        return min(importance, 1.0)
    
    def merge_similar_scenes(self, scenes: List[Dict], similarity_threshold: float = 0.8) -> List[Dict]:
        """
        Merge consecutive scenes that are very similar.
        
        Args:
            scenes: List of scene dictionaries
            similarity_threshold: Threshold for merging scenes (0-1)
            
        Returns:
            List of merged scenes
        """
        if len(scenes) <= 1:
            return scenes
        
        merged_scenes = []
        current_scene = scenes[0].copy()
        
        for next_scene in scenes[1:]:
            # Check if scenes should be merged based on duration and importance
            duration_diff = abs(current_scene['duration'] - next_scene['duration'])
            importance_diff = abs(
                current_scene.get('importance_score', 0.5) - 
                next_scene.get('importance_score', 0.5)
            )
            
            similarity = 1.0 - (duration_diff / 60.0 + importance_diff) / 2.0
            
            if similarity >= similarity_threshold and current_scene['duration'] < 10.0:
                # Merge scenes
                current_scene['end_time'] = next_scene['end_time']
                current_scene['duration'] = current_scene['end_time'] - current_scene['start_time']
                current_scene['end_frame'] = next_scene['end_frame']
                current_scene['frame_count'] = current_scene['end_frame'] - current_scene['start_frame']
                
                # Average importance scores
                current_importance = current_scene.get('importance_score', 0.5)
                next_importance = next_scene.get('importance_score', 0.5)
                current_scene['importance_score'] = (current_importance + next_importance) / 2.0
            else:
                # Add current scene and start new one
                merged_scenes.append(current_scene)
                current_scene = next_scene.copy()
        
        # Add final scene
        merged_scenes.append(current_scene)
        
        # Renumber scenes
        for i, scene in enumerate(merged_scenes):
            scene['scene_number'] = i + 1
        
        logger.info(f"Merged {len(scenes)} scenes into {len(merged_scenes)} scenes")
        return merged_scenes
    
    def get_scene_statistics(self, scenes: List[Dict]) -> Dict:
        """Calculate statistics about detected scenes."""
        if not scenes:
            return {}
        
        durations = [scene['duration'] for scene in scenes]
        importance_scores = [scene.get('importance_score', 0) for scene in scenes]
        
        stats = {
            'total_scenes': len(scenes),
            'total_duration': sum(durations),
            'average_duration': np.mean(durations),
            'median_duration': np.median(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'average_importance': np.mean(importance_scores),
            'high_importance_scenes': len([s for s in importance_scores if s > 0.7])
        }
        
        return stats
