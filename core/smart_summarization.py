"""
Smart summarization engine that combines multiple analysis methods.
Integrates visual, audio, and narrative analysis for optimal video summaries.
"""
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class SmartSummarizationEngine:
    """Intelligent summarization combining multiple analysis methods."""
    
    def __init__(self):
        """Initialize the smart summarization engine."""
        self.summary_algorithms = {
            'importance_based': self._importance_based_summary,
            'narrative_structure': self._narrative_structure_summary,
            'audio_visual_sync': self._audio_visual_sync_summary,
            'emotional_flow': self._emotional_flow_summary,
            'hybrid': self._hybrid_summary
        }
        
        self.target_compression_ratios = {
            'short': 0.05,    # 5% - Ultra short highlight reel
            'medium': 0.15,   # 15% - Standard summary
            'long': 0.25,     # 25% - Extended summary
            'custom': 0.15    # User-defined
        }
    
    def create_intelligent_summary(self, video_data: Dict, 
                                 summary_type: str = 'medium',
                                 algorithm: str = 'hybrid') -> Dict:
        """
        Create an intelligent summary using specified algorithm and compression.
        
        Args:
            video_data: Complete video analysis data
            summary_type: Target summary length (short/medium/long/custom)
            algorithm: Summarization algorithm to use
            
        Returns:
            Comprehensive summary with multiple output formats
        """
        try:
            target_ratio = self.target_compression_ratios.get(summary_type, 0.15)
            
            # Get enhanced scenes with all analysis data
            enhanced_scenes = video_data.get('enhanced_scenes', [])
            if not enhanced_scenes:
                logger.warning("No enhanced scenes found, using basic scenes")
                enhanced_scenes = video_data.get('scenes', [])
            
            # Apply selected summarization algorithm
            if algorithm in self.summary_algorithms:
                summary_result = self.summary_algorithms[algorithm](
                    enhanced_scenes, target_ratio, video_data
                )
            else:
                logger.warning(f"Unknown algorithm {algorithm}, using hybrid")
                summary_result = self._hybrid_summary(enhanced_scenes, target_ratio, video_data)
            
            # Enhance summary with metadata
            summary_result['metadata'] = self._generate_summary_metadata(
                video_data, summary_result, algorithm, summary_type
            )
            
            # Generate multiple output formats
            summary_result['outputs'] = self._generate_output_formats(
                summary_result, video_data
            )
            
            logger.info(f"Created {algorithm} summary with {target_ratio*100:.1f}% compression")
            return summary_result
            
        except Exception as e:
            logger.error(f"Error in intelligent summarization: {e}")
            return self._fallback_summary(video_data, target_ratio)
    
    def _importance_based_summary(self, scenes: List[Dict], target_ratio: float, 
                                video_data: Dict) -> Dict:
        """Create summary based purely on scene importance scores."""
        try:
            # Extract importance scores
            scene_importance = []
            for i, scene in enumerate(scenes):
                importance = scene.get('importance_score', 0)
                scene_importance.append((i, importance, scene))
            
            # Sort by importance
            scene_importance.sort(key=lambda x: x[1], reverse=True)
            
            # Select top scenes to meet target ratio
            total_duration = sum(
                scene.get('end_time', 0) - scene.get('start_time', 0) 
                for scene in scenes
            )
            target_duration = total_duration * target_ratio
            
            selected_scenes = []
            selected_duration = 0
            
            for scene_idx, importance, scene in scene_importance:
                scene_duration = scene.get('end_time', 0) - scene.get('start_time', 0)
                if selected_duration + scene_duration <= target_duration:
                    selected_scenes.append({
                        'original_index': scene_idx,
                        'scene_data': scene,
                        'selection_reason': f'High importance ({importance:.3f})'
                    })
                    selected_duration += scene_duration
            
            # Sort selected scenes back to chronological order
            selected_scenes.sort(key=lambda x: x['original_index'])
            
            return {
                'algorithm': 'importance_based',
                'selected_scenes': selected_scenes,
                'compression_ratio': selected_duration / total_duration,
                'selection_criteria': 'Scene importance scores',
                'total_selected_duration': selected_duration
            }
            
        except Exception as e:
            logger.error(f"Error in importance-based summary: {e}")
            return {'selected_scenes': [], 'compression_ratio': 0}
    
    def _narrative_structure_summary(self, scenes: List[Dict], target_ratio: float,
                                   video_data: Dict) -> Dict:
        """Create summary following narrative structure (5-act format)."""
        try:
            total_scenes = len(scenes)
            if total_scenes == 0:
                return {'selected_scenes': [], 'compression_ratio': 0}
            
            # Define narrative structure points
            narrative_points = {
                'opening': (0, int(total_scenes * 0.1)),           # First 10%
                'inciting_incident': (int(total_scenes * 0.1), int(total_scenes * 0.25)),  # 10-25%
                'rising_action': (int(total_scenes * 0.25), int(total_scenes * 0.5)),      # 25-50%
                'climax': (int(total_scenes * 0.5), int(total_scenes * 0.75)),             # 50-75%
                'resolution': (int(total_scenes * 0.75), total_scenes)                     # 75-100%
            }
            
            # Allocate target duration across narrative acts
            total_duration = sum(
                scene.get('end_time', 0) - scene.get('start_time', 0) 
                for scene in scenes
            )
            target_duration = total_duration * target_ratio
            
            # Distribute target duration: more weight to middle acts
            act_weights = {'opening': 0.15, 'inciting_incident': 0.2, 'rising_action': 0.3, 'climax': 0.25, 'resolution': 0.1}
            
            selected_scenes = []
            
            for act_name, (start_idx, end_idx) in narrative_points.items():
                act_target_duration = target_duration * act_weights[act_name]
                act_scenes = scenes[start_idx:end_idx]
                
                # Select best scenes from this act
                act_scene_importance = []
                for i, scene in enumerate(act_scenes):
                    importance = scene.get('importance_score', 0)
                    # Boost importance for certain scene types in specific acts
                    if act_name == 'climax' and scene.get('scene_type') == 'action':
                        importance *= 1.5
                    elif act_name == 'opening' and scene.get('scene_type') == 'establishing':
                        importance *= 1.3
                    
                    act_scene_importance.append((start_idx + i, importance, scene))
                
                # Sort by importance within act
                act_scene_importance.sort(key=lambda x: x[1], reverse=True)
                
                # Select scenes to meet act target duration
                act_selected_duration = 0
                for scene_idx, importance, scene in act_scene_importance:
                    scene_duration = scene.get('end_time', 0) - scene.get('start_time', 0)
                    if act_selected_duration + scene_duration <= act_target_duration:
                        selected_scenes.append({
                            'original_index': scene_idx,
                            'scene_data': scene,
                            'selection_reason': f'{act_name.title()} act scene (importance: {importance:.3f})'
                        })
                        act_selected_duration += scene_duration
            
            # Sort by chronological order
            selected_scenes.sort(key=lambda x: x['original_index'])
            
            total_selected_duration = sum(
                scene['scene_data'].get('end_time', 0) - scene['scene_data'].get('start_time', 0)
                for scene in selected_scenes
            )
            
            return {
                'algorithm': 'narrative_structure',
                'selected_scenes': selected_scenes,
                'compression_ratio': total_selected_duration / total_duration,
                'selection_criteria': '5-act narrative structure',
                'total_selected_duration': total_selected_duration
            }
            
        except Exception as e:
            logger.error(f"Error in narrative structure summary: {e}")
            return {'selected_scenes': [], 'compression_ratio': 0}
    
    def _audio_visual_sync_summary(self, scenes: List[Dict], target_ratio: float,
                                 video_data: Dict) -> Dict:
        """Create summary focusing on audio-visual synchronization and peak moments."""
        try:
            # Score scenes based on audio-visual characteristics
            scored_scenes = []
            
            for i, scene in enumerate(scenes):
                visual_analysis = scene.get('visual_analysis', {})
                audio_analysis = scene.get('audio_analysis', {})
                
                # Visual scoring factors
                motion_intensity = visual_analysis.get('motion_intensity', 0)
                cut_frequency = visual_analysis.get('cut_frequency', 0)
                visual_complexity = (motion_intensity + cut_frequency) / 2
                
                # Audio scoring factors
                volume_stats = audio_analysis.get('volume_stats', {})
                speech_indicators = audio_analysis.get('speech_indicators', {})
                music_indicators = audio_analysis.get('music_indicators', {})
                
                volume_score = volume_stats.get('rms_volume', 0)
                speech_score = speech_indicators.get('speech_probability', 0)
                music_score = music_indicators.get('music_probability', 0)
                
                # Calculate sync score (how well audio and visual complement each other)
                if motion_intensity > 0.5 and volume_score > 0.3:  # Action with sound
                    sync_score = 1.0
                elif motion_intensity < 0.2 and speech_score > 0.7:  # Dialogue scene
                    sync_score = 0.9
                elif cut_frequency > 0.6 and music_score > 0.6:  # Music montage
                    sync_score = 0.8
                else:
                    sync_score = 0.5
                
                # Combined score
                total_score = (
                    visual_complexity * 0.3 +
                    (volume_score + speech_score + music_score) * 0.4 +
                    sync_score * 0.3
                )
                
                scored_scenes.append((i, total_score, scene))
            
            # Sort by score
            scored_scenes.sort(key=lambda x: x[1], reverse=True)
            
            # Select scenes to meet target ratio
            total_duration = sum(
                scene.get('end_time', 0) - scene.get('start_time', 0) 
                for scene in scenes
            )
            target_duration = total_duration * target_ratio
            
            selected_scenes = []
            selected_duration = 0
            
            for scene_idx, score, scene in scored_scenes:
                scene_duration = scene.get('end_time', 0) - scene.get('start_time', 0)
                if selected_duration + scene_duration <= target_duration:
                    selected_scenes.append({
                        'original_index': scene_idx,
                        'scene_data': scene,
                        'selection_reason': f'Audio-visual sync score: {score:.3f}'
                    })
                    selected_duration += scene_duration
            
            # Sort chronologically
            selected_scenes.sort(key=lambda x: x['original_index'])
            
            return {
                'algorithm': 'audio_visual_sync',
                'selected_scenes': selected_scenes,
                'compression_ratio': selected_duration / total_duration,
                'selection_criteria': 'Audio-visual synchronization and peak moments',
                'total_selected_duration': selected_duration
            }
            
        except Exception as e:
            logger.error(f"Error in audio-visual sync summary: {e}")
            return {'selected_scenes': [], 'compression_ratio': 0}
    
    def _emotional_flow_summary(self, scenes: List[Dict], target_ratio: float,
                              video_data: Dict) -> Dict:
        """Create summary maintaining emotional flow and character development."""
        try:
            # Analyze emotional progression
            emotional_scenes = []
            
            for i, scene in enumerate(scenes):
                emotional_analysis = scene.get('emotional_analysis', {})
                
                # Get dominant emotion and intensity
                if emotional_analysis:
                    dominant_emotion = max(emotional_analysis, key=emotional_analysis.get)
                    emotion_intensity = emotional_analysis[dominant_emotion]
                else:
                    dominant_emotion = 'neutral'
                    emotion_intensity = 0.0
                
                emotional_scenes.append({
                    'index': i,
                    'scene': scene,
                    'emotion': dominant_emotion,
                    'intensity': emotion_intensity
                })
            
            # Identify emotional peaks and transitions
            selected_scenes = []
            
            # Always include high-intensity emotional moments
            high_intensity_scenes = [
                es for es in emotional_scenes 
                if es['intensity'] > 0.7
            ]
            
            # Include emotional transitions (when emotion changes significantly)
            transition_scenes = []
            for i in range(1, len(emotional_scenes)):
                current = emotional_scenes[i]
                previous = emotional_scenes[i-1]
                
                if (current['emotion'] != previous['emotion'] and 
                    current['intensity'] > 0.5):
                    transition_scenes.append(current)
            
            # Combine and sort by importance
            candidate_scenes = high_intensity_scenes + transition_scenes
            
            # Remove duplicates and sort by scene index
            seen_indices = set()
            unique_candidates = []
            for es in candidate_scenes:
                if es['index'] not in seen_indices:
                    unique_candidates.append(es)
                    seen_indices.add(es['index'])
            
            unique_candidates.sort(key=lambda x: x['index'])
            
            # Select scenes to meet target ratio
            total_duration = sum(
                scene.get('end_time', 0) - scene.get('start_time', 0) 
                for scene in scenes
            )
            target_duration = total_duration * target_ratio
            
            selected_duration = 0
            for es in unique_candidates:
                scene = es['scene']
                scene_duration = scene.get('end_time', 0) - scene.get('start_time', 0)
                
                if selected_duration + scene_duration <= target_duration:
                    selected_scenes.append({
                        'original_index': es['index'],
                        'scene_data': scene,
                        'selection_reason': f'Emotional {es["emotion"]} (intensity: {es["intensity"]:.3f})'
                    })
                    selected_duration += scene_duration
            
            return {
                'algorithm': 'emotional_flow',
                'selected_scenes': selected_scenes,
                'compression_ratio': selected_duration / total_duration,
                'selection_criteria': 'Emotional peaks and transitions',
                'total_selected_duration': selected_duration
            }
            
        except Exception as e:
            logger.error(f"Error in emotional flow summary: {e}")
            return {'selected_scenes': [], 'compression_ratio': 0}
    
    def _hybrid_summary(self, scenes: List[Dict], target_ratio: float,
                       video_data: Dict) -> Dict:
        """Create hybrid summary combining multiple algorithms."""
        try:
            # Run multiple algorithms with different weights
            algorithms = {
                'importance': (self._importance_based_summary, 0.4),
                'narrative': (self._narrative_structure_summary, 0.3),
                'audio_visual': (self._audio_visual_sync_summary, 0.2),
                'emotional': (self._emotional_flow_summary, 0.1)
            }
            
            # Collect scenes from all algorithms
            all_candidate_scenes = []
            
            for alg_name, (alg_func, weight) in algorithms.items():
                try:
                    result = alg_func(scenes, target_ratio, video_data)
                    for scene_info in result.get('selected_scenes', []):
                        scene_info['algorithm_source'] = alg_name
                        scene_info['algorithm_weight'] = weight
                        all_candidate_scenes.append(scene_info)
                except Exception as e:
                    logger.warning(f"Algorithm {alg_name} failed: {e}")
            
            # Score each unique scene based on how many algorithms selected it
            scene_scores = {}
            for scene_info in all_candidate_scenes:
                scene_idx = scene_info['original_index']
                weight = scene_info['algorithm_weight']
                
                if scene_idx not in scene_scores:
                    scene_scores[scene_idx] = {
                        'score': 0,
                        'scene_data': scene_info['scene_data'],
                        'sources': []
                    }
                
                scene_scores[scene_idx]['score'] += weight
                scene_scores[scene_idx]['sources'].append(scene_info['algorithm_source'])
            
            # Sort by combined score
            sorted_scenes = sorted(
                scene_scores.items(),
                key=lambda x: x[1]['score'],
                reverse=True
            )
            
            # Select scenes to meet target ratio
            total_duration = sum(
                scene.get('end_time', 0) - scene.get('start_time', 0) 
                for scene in scenes
            )
            target_duration = total_duration * target_ratio
            
            selected_scenes = []
            selected_duration = 0
            
            for scene_idx, scene_info in sorted_scenes:
                scene = scene_info['scene_data']
                scene_duration = scene.get('end_time', 0) - scene.get('start_time', 0)
                
                if selected_duration + scene_duration <= target_duration:
                    selected_scenes.append({
                        'original_index': scene_idx,
                        'scene_data': scene,
                        'selection_reason': f'Hybrid score: {scene_info["score"]:.3f} (from: {", ".join(scene_info["sources"])})'
                    })
                    selected_duration += scene_duration
            
            # Sort chronologically
            selected_scenes.sort(key=lambda x: x['original_index'])
            
            return {
                'algorithm': 'hybrid',
                'selected_scenes': selected_scenes,
                'compression_ratio': selected_duration / total_duration,
                'selection_criteria': 'Hybrid multi-algorithm approach',
                'total_selected_duration': selected_duration
            }
            
        except Exception as e:
            logger.error(f"Error in hybrid summary: {e}")
            return self._fallback_summary(video_data, target_ratio)
    
    def _generate_summary_metadata(self, video_data: Dict, summary_result: Dict,
                                 algorithm: str, summary_type: str) -> Dict:
        """Generate comprehensive metadata for the summary."""
        return {
            'algorithm_used': algorithm,
            'summary_type': summary_type,
            'target_compression': self.target_compression_ratios.get(summary_type, 0.15),
            'actual_compression': summary_result.get('compression_ratio', 0),
            'total_scenes_available': len(video_data.get('enhanced_scenes', [])),
            'scenes_selected': len(summary_result.get('selected_scenes', [])),
            'selection_criteria': summary_result.get('selection_criteria', ''),
            'processing_timestamp': str(np.datetime64('now')),
            'video_title': video_data.get('title', 'Unknown'),
            'original_duration': video_data.get('duration', 0),
            'summary_duration': summary_result.get('total_selected_duration', 0)
        }
    
    def _generate_output_formats(self, summary_result: Dict, video_data: Dict) -> Dict:
        """Generate multiple output formats for the summary."""
        outputs = {}
        
        try:
            selected_scenes = summary_result.get('selected_scenes', [])
            
            # Chapter-based breakdown
            chapters = []
            for i, scene_info in enumerate(selected_scenes):
                scene = scene_info['scene_data']
                chapters.append({
                    'chapter_number': i + 1,
                    'title': f"Scene {scene_info['original_index'] + 1}",
                    'start_time': scene.get('start_time', 0),
                    'end_time': scene.get('end_time', 0),
                    'duration': scene.get('end_time', 0) - scene.get('start_time', 0),
                    'description': scene_info.get('selection_reason', ''),
                    'scene_type': scene.get('scene_type', 'unknown')
                })
            
            outputs['chapters'] = chapters
            
            # Timestamp list for video editors
            timestamps = [
                {
                    'start': scene_info['scene_data'].get('start_time', 0),
                    'end': scene_info['scene_data'].get('end_time', 0),
                    'label': f"Key Scene {i+1}",
                    'reason': scene_info.get('selection_reason', '')
                }
                for i, scene_info in enumerate(selected_scenes)
            ]
            
            outputs['timestamps'] = timestamps
            
            # Summary statistics
            outputs['statistics'] = {
                'original_length_minutes': video_data.get('duration', 0) / 60,
                'summary_length_minutes': summary_result.get('total_selected_duration', 0) / 60,
                'compression_percentage': summary_result.get('compression_ratio', 0) * 100,
                'scenes_included': len(selected_scenes),
                'average_scene_length': (
                    summary_result.get('total_selected_duration', 0) / len(selected_scenes)
                    if selected_scenes else 0
                )
            }
            
        except Exception as e:
            logger.error(f"Error generating output formats: {e}")
        
        return outputs
    
    def _fallback_summary(self, video_data: Dict, target_ratio: float) -> Dict:
        """Fallback summary method when other algorithms fail."""
        scenes = video_data.get('scenes', [])
        if not scenes:
            return {'selected_scenes': [], 'compression_ratio': 0}
        
        # Simple selection: take every Nth scene
        total_scenes = len(scenes)
        target_scenes = max(1, int(total_scenes * target_ratio))
        step = max(1, total_scenes // target_scenes)
        
        selected_scenes = []
        for i in range(0, total_scenes, step):
            if len(selected_scenes) < target_scenes:
                selected_scenes.append({
                    'original_index': i,
                    'scene_data': scenes[i],
                    'selection_reason': 'Fallback uniform sampling'
                })
        
        total_duration = sum(
            scene.get('end_time', 0) - scene.get('start_time', 0) 
            for scene in scenes
        )
        selected_duration = sum(
            scene['scene_data'].get('end_time', 0) - scene['scene_data'].get('start_time', 0)
            for scene in selected_scenes
        )
        
        return {
            'algorithm': 'fallback',
            'selected_scenes': selected_scenes,
            'compression_ratio': selected_duration / total_duration if total_duration > 0 else 0,
            'selection_criteria': 'Uniform sampling fallback',
            'total_selected_duration': selected_duration
        }