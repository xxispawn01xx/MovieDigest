"""
Advanced scene analysis for enhanced video summarization.
Provides detailed scene characterization, emotional analysis, and narrative flow detection.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from scipy import stats
import json

logger = logging.getLogger(__name__)

class AdvancedSceneAnalyzer:
    """Advanced scene analysis with visual and audio characteristics."""
    
    def __init__(self):
        """Initialize the advanced scene analyzer."""
        self.scene_types = {
            'action': {'motion_threshold': 0.7, 'cut_frequency': 0.8},
            'dialogue': {'motion_threshold': 0.2, 'cut_frequency': 0.3},
            'establishing': {'motion_threshold': 0.3, 'cut_frequency': 0.2},
            'transition': {'motion_threshold': 0.4, 'cut_frequency': 0.5},
            'emotional': {'motion_threshold': 0.1, 'cut_frequency': 0.4},
            'montage': {'motion_threshold': 0.6, 'cut_frequency': 0.9}
        }
        
        self.emotional_indicators = {
            'tension': ['fast_cuts', 'high_motion', 'color_contrast'],
            'calm': ['slow_cuts', 'low_motion', 'warm_colors'],
            'excitement': ['fast_cuts', 'high_motion', 'bright_colors'],
            'sadness': ['slow_cuts', 'low_motion', 'cool_colors'],
            'suspense': ['medium_cuts', 'low_motion', 'dark_colors']
        }
    
    def analyze_scene_characteristics(self, video_path: str, scene_data: List[Dict]) -> List[Dict]:
        """
        Analyze detailed characteristics of each scene.
        
        Args:
            video_path: Path to video file
            scene_data: List of scene dictionaries with start/end times
            
        Returns:
            Enhanced scene data with detailed analysis
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Cannot open video: {video_path}")
                return scene_data
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            enhanced_scenes = []
            
            for i, scene in enumerate(scene_data):
                start_time = scene.get('start_time', 0)
                end_time = scene.get('end_time', start_time + 30)
                
                # Analyze visual characteristics
                visual_analysis = self._analyze_visual_characteristics(
                    cap, start_time, end_time, fps
                )
                
                # Determine scene type based on characteristics
                scene_type = self._classify_scene_type(visual_analysis)
                
                # Analyze emotional content
                emotional_analysis = self._analyze_emotional_content(visual_analysis)
                
                # Calculate scene importance
                importance_score = self._calculate_scene_importance(
                    visual_analysis, emotional_analysis, i, len(scene_data)
                )
                
                # Enhanced scene data
                enhanced_scene = {
                    **scene,
                    'scene_type': scene_type,
                    'visual_analysis': visual_analysis,
                    'emotional_analysis': emotional_analysis,
                    'importance_score': importance_score,
                    'characteristics': self._generate_scene_characteristics(
                        visual_analysis, emotional_analysis
                    )
                }
                
                enhanced_scenes.append(enhanced_scene)
                
                # Progress logging
                if (i + 1) % 10 == 0:
                    logger.info(f"Analyzed {i + 1}/{len(scene_data)} scenes")
            
            cap.release()
            logger.info(f"Completed advanced analysis of {len(enhanced_scenes)} scenes")
            return enhanced_scenes
            
        except Exception as e:
            logger.error(f"Error in advanced scene analysis: {e}")
            return scene_data
    
    def _analyze_visual_characteristics(self, cap: cv2.VideoCapture, 
                                      start_time: float, end_time: float, 
                                      fps: float) -> Dict:
        """Analyze visual characteristics of a scene."""
        characteristics = {
            'motion_intensity': 0.0,
            'cut_frequency': 0.0,
            'brightness_avg': 0.0,
            'contrast_avg': 0.0,
            'color_dominance': {},
            'frame_variance': 0.0,
            'edge_density': 0.0
        }
        
        try:
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            frame_count = end_frame - start_frame
            
            if frame_count <= 0:
                return characteristics
            
            # Sample frames for analysis
            sample_frames = min(30, frame_count)  # Limit to 30 frames for performance
            frame_indices = np.linspace(start_frame, end_frame - 1, sample_frames, dtype=int)
            
            frames_data = []
            previous_frame = None
            motion_values = []
            brightness_values = []
            contrast_values = []
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Convert to grayscale for analysis
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames_data.append(gray)
                
                # Calculate brightness and contrast
                brightness = np.mean(gray)
                contrast = np.std(gray)
                brightness_values.append(brightness)
                contrast_values.append(contrast)
                
                # Calculate motion between consecutive frames
                if previous_frame is not None:
                    motion = self._calculate_frame_motion(previous_frame, gray)
                    motion_values.append(motion)
                
                previous_frame = gray
            
            # Aggregate characteristics
            if motion_values:
                characteristics['motion_intensity'] = np.mean(motion_values)
            
            if brightness_values:
                characteristics['brightness_avg'] = np.mean(brightness_values)
            
            if contrast_values:
                characteristics['contrast_avg'] = np.mean(contrast_values)
            
            # Calculate cut frequency (scene changes)
            if len(motion_values) > 1:
                motion_threshold = np.percentile(motion_values, 75)
                cuts = sum(1 for m in motion_values if m > motion_threshold)
                characteristics['cut_frequency'] = cuts / len(motion_values)
            
            # Calculate frame variance
            if len(frames_data) > 1:
                frame_diffs = []
                for i in range(1, len(frames_data)):
                    diff = np.mean(np.abs(frames_data[i] - frames_data[i-1]))
                    frame_diffs.append(diff)
                characteristics['frame_variance'] = np.mean(frame_diffs) if frame_diffs else 0.0
            
            # Analyze color dominance (using original color frames)
            characteristics['color_dominance'] = self._analyze_color_dominance(cap, frame_indices)
            
        except Exception as e:
            logger.error(f"Error analyzing visual characteristics: {e}")
        
        return characteristics
    
    def _calculate_frame_motion(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate motion between two frames using optical flow."""
        try:
            # Resize frames for faster computation
            h, w = frame1.shape
            if h > 480:
                scale = 480 / h
                new_h, new_w = int(h * scale), int(w * scale)
                frame1 = cv2.resize(frame1, (new_w, new_h))
                frame2 = cv2.resize(frame2, (new_w, new_h))
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowPyrLK(
                frame1, frame2, 
                np.array([[x, y] for y in range(0, frame1.shape[0], 20) 
                         for x in range(0, frame1.shape[1], 20)], dtype=np.float32).reshape(-1, 1, 2),
                None
            )[0]
            
            if flow is not None and len(flow) > 0:
                # Calculate magnitude of motion vectors
                motion_magnitude = np.sqrt(flow[:, 0, 0]**2 + flow[:, 0, 1]**2)
                return np.mean(motion_magnitude)
            
            return 0.0
            
        except Exception:
            # Fallback to simple frame difference
            diff = cv2.absdiff(frame1, frame2)
            return np.mean(diff) / 255.0
    
    def _analyze_color_dominance(self, cap: cv2.VideoCapture, frame_indices: np.ndarray) -> Dict:
        """Analyze dominant colors in the scene."""
        color_analysis = {
            'warm_ratio': 0.0,
            'cool_ratio': 0.0,
            'saturation_avg': 0.0,
            'dominant_hue': 0
        }
        
        try:
            hue_values = []
            saturation_values = []
            warm_pixels = 0
            cool_pixels = 0
            total_pixels = 0
            
            # Sample fewer frames for color analysis
            sample_indices = frame_indices[::max(1, len(frame_indices) // 5)]
            
            for frame_idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Convert to HSV for better color analysis
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                
                # Sample pixels for analysis (every 10th pixel)
                h_sample = hsv[::10, ::10, 0]
                s_sample = hsv[::10, ::10, 1]
                
                hue_values.extend(h_sample.flatten())
                saturation_values.extend(s_sample.flatten())
                
                # Count warm vs cool pixels
                warm_mask = ((h_sample < 30) | (h_sample > 150))
                cool_mask = ((h_sample >= 60) & (h_sample <= 120))
                
                warm_pixels += np.sum(warm_mask)
                cool_pixels += np.sum(cool_mask)
                total_pixels += h_sample.size
            
            if total_pixels > 0:
                color_analysis['warm_ratio'] = warm_pixels / total_pixels
                color_analysis['cool_ratio'] = cool_pixels / total_pixels
            
            if hue_values:
                color_analysis['dominant_hue'] = int(stats.mode(hue_values)[0])
            
            if saturation_values:
                color_analysis['saturation_avg'] = np.mean(saturation_values)
        
        except Exception as e:
            logger.error(f"Error in color analysis: {e}")
        
        return color_analysis
    
    def _classify_scene_type(self, visual_analysis: Dict) -> str:
        """Classify scene type based on visual characteristics."""
        motion = visual_analysis.get('motion_intensity', 0)
        cuts = visual_analysis.get('cut_frequency', 0)
        
        # Calculate similarity scores for each scene type
        type_scores = {}
        
        for scene_type, thresholds in self.scene_types.items():
            motion_score = 1 - abs(motion - thresholds['motion_threshold'])
            cut_score = 1 - abs(cuts - thresholds['cut_frequency'])
            type_scores[scene_type] = (motion_score + cut_score) / 2
        
        # Return the scene type with highest score
        return max(type_scores, key=type_scores.get)
    
    def _analyze_emotional_content(self, visual_analysis: Dict) -> Dict:
        """Analyze emotional content of the scene."""
        emotional_scores = {}
        
        motion = visual_analysis.get('motion_intensity', 0)
        cuts = visual_analysis.get('cut_frequency', 0)
        brightness = visual_analysis.get('brightness_avg', 128) / 255.0
        contrast = visual_analysis.get('contrast_avg', 0) / 255.0
        warm_ratio = visual_analysis.get('color_dominance', {}).get('warm_ratio', 0)
        
        # Score each emotion based on visual indicators
        emotional_scores['tension'] = (cuts * 0.4 + motion * 0.4 + contrast * 0.2)
        emotional_scores['calm'] = ((1 - cuts) * 0.4 + (1 - motion) * 0.4 + warm_ratio * 0.2)
        emotional_scores['excitement'] = (cuts * 0.3 + motion * 0.4 + brightness * 0.3)
        emotional_scores['sadness'] = ((1 - brightness) * 0.5 + (1 - warm_ratio) * 0.3 + (1 - motion) * 0.2)
        emotional_scores['suspense'] = (contrast * 0.4 + (1 - brightness) * 0.4 + motion * 0.2)
        
        # Normalize scores
        max_score = max(emotional_scores.values()) if emotional_scores.values() else 1
        if max_score > 0:
            emotional_scores = {k: v / max_score for k, v in emotional_scores.items()}
        
        return emotional_scores
    
    def _calculate_scene_importance(self, visual_analysis: Dict, 
                                  emotional_analysis: Dict, 
                                  scene_index: int, total_scenes: int) -> float:
        """Calculate the importance score of a scene."""
        importance_factors = []
        
        # Visual complexity factor
        motion = visual_analysis.get('motion_intensity', 0)
        cuts = visual_analysis.get('cut_frequency', 0)
        contrast = visual_analysis.get('contrast_avg', 0) / 255.0
        
        visual_complexity = (motion + cuts + contrast) / 3
        importance_factors.append(visual_complexity * 0.3)
        
        # Emotional intensity factor
        max_emotion = max(emotional_analysis.values()) if emotional_analysis.values() else 0
        importance_factors.append(max_emotion * 0.4)
        
        # Position factor (beginning and end scenes are often important)
        position_factor = 0.2
        if scene_index < total_scenes * 0.1:  # First 10%
            position_factor = 0.8
        elif scene_index > total_scenes * 0.9:  # Last 10%
            position_factor = 0.6
        elif scene_index > total_scenes * 0.4 and scene_index < total_scenes * 0.6:  # Middle
            position_factor = 0.5
        
        importance_factors.append(position_factor * 0.3)
        
        # Calculate final importance score
        importance_score = sum(importance_factors)
        return min(1.0, max(0.0, importance_score))
    
    def _generate_scene_characteristics(self, visual_analysis: Dict, 
                                      emotional_analysis: Dict) -> List[str]:
        """Generate human-readable characteristics for the scene."""
        characteristics = []
        
        # Motion characteristics
        motion = visual_analysis.get('motion_intensity', 0)
        if motion > 0.6:
            characteristics.append("high_action")
        elif motion < 0.2:
            characteristics.append("static")
        else:
            characteristics.append("moderate_motion")
        
        # Cut frequency characteristics
        cuts = visual_analysis.get('cut_frequency', 0)
        if cuts > 0.7:
            characteristics.append("fast_paced")
        elif cuts < 0.3:
            characteristics.append("slow_paced")
        
        # Brightness characteristics
        brightness = visual_analysis.get('brightness_avg', 128) / 255.0
        if brightness > 0.7:
            characteristics.append("bright")
        elif brightness < 0.3:
            characteristics.append("dark")
        
        # Color characteristics
        color_dom = visual_analysis.get('color_dominance', {})
        warm_ratio = color_dom.get('warm_ratio', 0)
        if warm_ratio > 0.6:
            characteristics.append("warm_tones")
        elif warm_ratio < 0.3:
            characteristics.append("cool_tones")
        
        # Emotional characteristics
        dominant_emotion = max(emotional_analysis, key=emotional_analysis.get) if emotional_analysis else None
        if dominant_emotion and emotional_analysis[dominant_emotion] > 0.6:
            characteristics.append(f"emotionally_{dominant_emotion}")
        
        return characteristics
    
    def generate_scene_summary(self, enhanced_scenes: List[Dict]) -> Dict:
        """Generate a comprehensive summary of all scene analysis."""
        if not enhanced_scenes:
            return {}
        
        summary = {
            'total_scenes': len(enhanced_scenes),
            'scene_type_distribution': {},
            'emotional_flow': [],
            'high_importance_scenes': [],
            'visual_characteristics': {
                'avg_motion_intensity': 0,
                'avg_cut_frequency': 0,
                'predominant_mood': '',
                'visual_complexity_score': 0
            },
            'narrative_structure': {
                'act_boundaries': [],
                'climax_scenes': [],
                'transition_points': []
            }
        }
        
        # Analyze scene type distribution
        scene_types = [scene.get('scene_type', 'unknown') for scene in enhanced_scenes]
        for scene_type in set(scene_types):
            summary['scene_type_distribution'][scene_type] = scene_types.count(scene_type)
        
        # Identify high importance scenes
        importance_scores = [(i, scene.get('importance_score', 0)) 
                           for i, scene in enumerate(enhanced_scenes)]
        importance_scores.sort(key=lambda x: x[1], reverse=True)
        
        summary['high_importance_scenes'] = [
            {
                'scene_index': idx,
                'importance_score': score,
                'start_time': enhanced_scenes[idx].get('start_time', 0),
                'scene_type': enhanced_scenes[idx].get('scene_type', 'unknown')
            }
            for idx, score in importance_scores[:10]
        ]
        
        # Calculate visual characteristics
        motion_values = [scene.get('visual_analysis', {}).get('motion_intensity', 0) 
                        for scene in enhanced_scenes]
        cut_values = [scene.get('visual_analysis', {}).get('cut_frequency', 0) 
                     for scene in enhanced_scenes]
        
        if motion_values:
            summary['visual_characteristics']['avg_motion_intensity'] = np.mean(motion_values)
        if cut_values:
            summary['visual_characteristics']['avg_cut_frequency'] = np.mean(cut_values)
        
        # Analyze emotional flow
        for i, scene in enumerate(enhanced_scenes):
            emotional_analysis = scene.get('emotional_analysis', {})
            if emotional_analysis:
                dominant_emotion = max(emotional_analysis, key=emotional_analysis.get)
                summary['emotional_flow'].append({
                    'scene_index': i,
                    'dominant_emotion': dominant_emotion,
                    'intensity': emotional_analysis[dominant_emotion]
                })
        
        # Identify narrative structure
        summary['narrative_structure'] = self._identify_narrative_structure(enhanced_scenes)
        
        return summary
    
    def _identify_narrative_structure(self, enhanced_scenes: List[Dict]) -> Dict:
        """Identify narrative structure elements in the scenes."""
        structure = {
            'act_boundaries': [],
            'climax_scenes': [],
            'transition_points': []
        }
        
        try:
            total_scenes = len(enhanced_scenes)
            
            # Identify potential act boundaries based on importance and scene type changes
            importance_scores = [scene.get('importance_score', 0) for scene in enhanced_scenes]
            scene_types = [scene.get('scene_type', 'unknown') for scene in enhanced_scenes]
            
            # Find peaks in importance that could indicate act boundaries
            if len(importance_scores) > 10:
                # Look for scenes with high importance at typical act positions
                act_positions = [
                    int(total_scenes * 0.25),  # End of Act 1
                    int(total_scenes * 0.5),   # Middle/Act 2
                    int(total_scenes * 0.75)   # Start of Act 3
                ]
                
                for pos in act_positions:
                    if pos < len(enhanced_scenes):
                        structure['act_boundaries'].append({
                            'scene_index': pos,
                            'importance_score': importance_scores[pos],
                            'estimated_act': len(structure['act_boundaries']) + 1
                        })
            
            # Identify climax scenes (highest importance in last third)
            last_third_start = int(total_scenes * 0.67)
            if last_third_start < total_scenes:
                climax_candidates = []
                for i in range(last_third_start, total_scenes):
                    climax_candidates.append((i, importance_scores[i]))
                
                # Sort by importance and take top 3
                climax_candidates.sort(key=lambda x: x[1], reverse=True)
                structure['climax_scenes'] = [
                    {
                        'scene_index': idx,
                        'importance_score': score
                    }
                    for idx, score in climax_candidates[:3]
                ]
            
            # Identify transition points (scene type changes)
            for i in range(1, len(scene_types)):
                if scene_types[i] != scene_types[i-1]:
                    structure['transition_points'].append({
                        'scene_index': i,
                        'from_type': scene_types[i-1],
                        'to_type': scene_types[i]
                    })
        
        except Exception as e:
            logger.error(f"Error identifying narrative structure: {e}")
        
        return structure