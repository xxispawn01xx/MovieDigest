"""
Video summarization module that combines scene detection, transcription, and narrative analysis
to create intelligent movie summaries with VLC bookmarks.
"""
import cv2
import numpy as np
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
import json
from datetime import datetime

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoSummarizer:
    """Main video summarization engine combining all analysis components."""
    
    def __init__(self):
        """Initialize video summarizer."""
        self.target_compression = config.SUMMARY_LENGTH_PERCENT / 100.0
        self.max_summary_duration = config.MAX_SUMMARY_LENGTH_MINUTES * 60
        self.min_summary_duration = config.MIN_SUMMARY_LENGTH_MINUTES * 60
    
    def create_summary(self, video_path: Path, scenes: List[Dict], 
                      transcription_data: List[Dict], 
                      narrative_analysis: Dict) -> Dict:
        """
        Create comprehensive video summary.
        
        Args:
            video_path: Path to original video
            scenes: Scene detection data
            transcription_data: Transcription segments
            narrative_analysis: Narrative analysis results
            
        Returns:
            Summary creation results
        """
        if isinstance(video_path, str):
            video_path = Path(video_path)
        logger.info(f"Creating summary for: {video_path.name}")
        
        try:
            # Calculate original duration
            original_duration = self._get_video_duration(video_path)
            target_duration = min(
                original_duration * self.target_compression,
                self.max_summary_duration
            )
            target_duration = max(target_duration, self.min_summary_duration)
            
            logger.info(f"Target summary duration: {target_duration/60:.1f} minutes")
            
            # Select scenes for summary
            selected_scenes = self._select_summary_scenes(
                scenes, target_duration, narrative_analysis
            )
            
            # Create video summary
            summary_path = self._create_video_summary(video_path, selected_scenes)
            
            # Generate VLC bookmarks
            bookmark_path = self._create_vlc_bookmarks(
                video_path, selected_scenes, narrative_analysis
            )
            
            # Create scene-by-scene breakdown
            scene_breakdown = self._create_scene_breakdown(
                selected_scenes, transcription_data, narrative_analysis
            )
            
            # Calculate summary statistics
            actual_duration = sum(scene['duration'] for scene in selected_scenes)
            compression_ratio = actual_duration / original_duration
            
            summary_result = {
                'summary_path': str(summary_path),
                'bookmark_path': str(bookmark_path),
                'selected_scenes': selected_scenes,
                'scene_breakdown': scene_breakdown,
                'original_duration': original_duration,
                'summary_length_seconds': actual_duration,
                'compression_ratio': compression_ratio,
                'target_duration': target_duration,
                'scene_count': len(selected_scenes),
                'created_at': datetime.now().isoformat()
            }
            
            logger.info(f"Summary created: {actual_duration/60:.1f}min "
                       f"({compression_ratio*100:.1f}% compression)")
            
            return summary_result
            
        except Exception as e:
            logger.error(f"Summary creation failed: {e}")
            raise
    
    def _select_summary_scenes(self, scenes: List[Dict], target_duration: float,
                              narrative_analysis: Dict) -> List[Dict]:
        """
        Select scenes for summary using importance scoring and duration constraints.
        
        Args:
            scenes: All detected scenes
            target_duration: Target summary duration in seconds
            narrative_analysis: Narrative analysis data
            
        Returns:
            List of selected scenes for summary
        """
        logger.info(f"Selecting scenes for {target_duration/60:.1f} minute summary")
        
        # Validate scenes have required fields
        valid_scenes = []
        for scene in scenes:
            if all(key in scene for key in ['start_time', 'duration', 'scene_number']):
                valid_scenes.append(scene)
            else:
                logger.warning(f"Skipping invalid scene: {scene}")
        
        if not valid_scenes:
            logger.error("No valid scenes found for summarization")
            raise RuntimeError("No valid scenes available - check scene detection output")
        
        logger.info(f"Processing {len(valid_scenes)} valid scenes")
        
        # Combine all importance scores
        scored_scenes = []
        for scene in valid_scenes:
            combined_score = self._calculate_combined_importance(scene, narrative_analysis)
            scored_scenes.append({
                **scene,
                'combined_importance': combined_score
            })
        
        # Sort by importance
        scored_scenes.sort(key=lambda x: x['combined_importance'], reverse=True)
        
        # Select scenes using strict duration constraints to prevent full copies
        selected_scenes = []
        total_duration = 0.0
        
        # Enforce strict 15% compression ratio from config
        logger.info(f"Target compression: {self.target_compression*100:.1f}%")
        logger.info(f"Target duration: {target_duration/60:.1f} minutes")
        
        # Only select scenes that fit within target duration
        for scene in scored_scenes:
            if total_duration + scene['duration'] <= target_duration:
                selected_scenes.append(scene)
                total_duration += scene['duration']
                logger.debug(f"Added scene {scene['scene_number']}: +{scene['duration']:.1f}s (total: {total_duration/60:.1f}min)")
            else:
                # Try to fit a shorter version of the scene if possible
                remaining_time = target_duration - total_duration
                if remaining_time > 5:  # At least 5 seconds worth including
                    truncated_scene = scene.copy()
                    truncated_scene['duration'] = remaining_time
                    truncated_scene['end_time'] = truncated_scene['start_time'] + remaining_time
                    selected_scenes.append(truncated_scene)
                    total_duration += remaining_time
                    logger.debug(f"Added truncated scene {scene['scene_number']}: +{remaining_time:.1f}s")
                break
        
        # Skip narrative flow enhancement to maintain strict compression ratio
        # selected_scenes = self._ensure_narrative_flow(selected_scenes, scenes, target_duration)
        
        # Sort by chronological order
        selected_scenes.sort(key=lambda x: x['start_time'])
        
        final_duration = sum(scene['duration'] for scene in selected_scenes)
        actual_compression = final_duration / (sum(scene['duration'] for scene in scenes) or 1)
        logger.info(f"Selected {len(selected_scenes)} scenes totaling {final_duration/60:.1f} minutes")
        logger.info(f"Achieved compression ratio: {actual_compression*100:.1f}%")
        
        return selected_scenes
    
    def _calculate_combined_importance(self, scene: Dict, narrative_analysis: Dict) -> float:
        """Calculate combined importance score from all factors."""
        # Base importance from scene detection
        base_score = scene.get('importance_score', 0.5)
        
        # Narrative importance from LLM analysis
        narrative_score = scene.get('narrative_importance', 0.5)
        
        # Position-based importance
        position_score = self._calculate_position_importance(scene, narrative_analysis)
        
        # Key moments bonus
        key_moments_score = self._calculate_key_moments_score(scene, narrative_analysis)
        
        # Duration-based scoring (prefer medium-length scenes)
        duration_score = self._calculate_duration_score(scene['duration'])
        
        # Weighted combination
        combined_score = (
            base_score * 0.25 +
            narrative_score * 0.35 +
            position_score * 0.15 +
            key_moments_score * 0.15 +
            duration_score * 0.10
        )
        
        return min(combined_score, 1.0)
    
    def _calculate_position_importance(self, scene: Dict, narrative_analysis: Dict) -> float:
        """Calculate importance based on narrative position."""
        # Implement three-act structure weighting
        structure = narrative_analysis.get('structure_analysis', {})
        
        # Simple position-based scoring
        scene_number = scene['scene_number']
        
        # Opening scenes (setup)
        if scene_number <= 3:
            return 0.8
        # Climax area (likely in final third)
        elif scene_number >= scene['scene_number'] * 0.7:
            return 0.9
        # Middle development
        else:
            return 0.6
    
    def _calculate_key_moments_score(self, scene: Dict, narrative_analysis: Dict) -> float:
        """Calculate bonus score for key narrative moments."""
        key_moments = narrative_analysis.get('key_moments', [])
        
        for moment in key_moments:
            if (moment.get('scene_number') == scene['scene_number'] or
                (moment.get('timestamp') and 
                 scene['start_time'] <= moment['timestamp'] <= scene['end_time'])):
                return 1.0
        
        return 0.3
    
    def _calculate_duration_score(self, duration: float) -> float:
        """Calculate score based on scene duration (prefer medium-length scenes)."""
        if duration < 5:
            return 0.3  # Too short
        elif duration > 120:
            return 0.4  # Too long
        elif 10 <= duration <= 60:
            return 1.0  # Ideal length
        else:
            return 0.7  # Acceptable length
    
    def _ensure_narrative_flow(self, selected_scenes: List[Dict], 
                              all_scenes: List[Dict], target_duration: float) -> List[Dict]:
        """Ensure selected scenes maintain narrative flow."""
        if len(selected_scenes) <= 1:
            return selected_scenes
        
        # Sort selected scenes by start time
        selected_scenes.sort(key=lambda x: x['start_time'])
        
        # Check for large gaps and add connecting scenes if needed
        enhanced_scenes = [selected_scenes[0]]
        current_duration = selected_scenes[0]['duration']
        
        for i in range(1, len(selected_scenes)):
            prev_scene = enhanced_scenes[-1]
            current_scene = selected_scenes[i]
            
            # Check gap between scenes
            gap_duration = current_scene['start_time'] - prev_scene['end_time']
            
            # If gap is large, try to add a connecting scene
            if gap_duration > 300 and current_duration < target_duration * 0.9:  # 5 minute gap
                connecting_scene = self._find_connecting_scene(
                    prev_scene, current_scene, all_scenes
                )
                
                if (connecting_scene and 
                    current_duration + connecting_scene['duration'] < target_duration):
                    enhanced_scenes.append(connecting_scene)
                    current_duration += connecting_scene['duration']
            
            enhanced_scenes.append(current_scene)
            current_duration += current_scene['duration']
        
        return enhanced_scenes
    
    def _find_connecting_scene(self, prev_scene: Dict, next_scene: Dict, 
                              all_scenes: List[Dict]) -> Optional[Dict]:
        """Find a scene that connects two selected scenes."""
        # Look for scenes between the two selected scenes
        connecting_scenes = [
            scene for scene in all_scenes
            if (prev_scene['end_time'] < scene['start_time'] < next_scene['start_time'] and
                scene['duration'] < 60)  # Short connecting scenes only
        ]
        
        if connecting_scenes:
            # Return the scene with highest importance in the gap
            return max(connecting_scenes, key=lambda x: x.get('importance_score', 0))
        
        return None
    
    def _create_video_summary(self, video_path: Path, selected_scenes: List[Dict]) -> Path:
        """Create the actual video summary file."""
        # Use custom output directory if set in session state
        import streamlit as st
        if hasattr(st, 'session_state') and hasattr(st.session_state, 'custom_output_dir') and st.session_state.custom_output_dir:
            base_output_dir = Path(st.session_state.custom_output_dir)
        else:
            base_output_dir = config.OUTPUT_DIR
            
        output_dir = base_output_dir / "summaries"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = output_dir / f"{video_path.stem}_summary_{timestamp}.mp4"
        
        logger.info(f"Creating video summary: {summary_path.name}")
        
        # Create ffmpeg filter for concatenating scenes
        filter_parts = []
        input_args = []
        
        for i, scene in enumerate(selected_scenes):
            start_time = scene['start_time']
            duration = scene['duration']
            
            input_args.extend(['-ss', str(start_time), '-t', str(duration), '-i', str(video_path)])
            filter_parts.append(f"[{i}:v][{i}:a]")
        
        # Validate scene selection
        if len(selected_scenes) == 0:
            logger.error("No scenes selected for summary - this should not happen")
            raise RuntimeError("Scene selection failed - no scenes available for summary")
        
        # Create a simple concatenation using individual clips
        cmd = ['ffmpeg']
        
        # Add input files for each scene
        for scene in selected_scenes:
            cmd.extend([
                '-ss', str(scene['start_time']),
                '-t', str(scene['duration']),
                '-i', str(video_path)
            ])
        
        # Fixed concatenation approach for better seeking and playback
        if len(selected_scenes) == 1:
            # Single scene - direct encoding
            cmd.extend([
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-movflags', '+faststart',  # Enable seeking
                '-reset_timestamps', '1',   # Reset timestamps for proper seeking
                '-y',
                str(summary_path)
            ])
        else:
            # Multiple scenes - use direct filter_complex for better seeking
            logger.info(f"Concatenating {len(selected_scenes)} scenes using filter_complex")
            
            # Build filter_complex for concatenation
            filter_str = ''.join([f'[{i}:v][{i}:a]' for i in range(len(selected_scenes))])
            filter_str += f'concat=n={len(selected_scenes)}:v=1:a=1[outv][outa]'
            
            cmd.extend([
                '-filter_complex', filter_str,
                '-map', '[outv]',
                '-map', '[outa]',
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-preset', 'fast',  # Faster encoding
                '-crf', '23',  # Good quality balance
                '-movflags', '+faststart',  # Enable seeking
                '-avoid_negative_ts', 'make_zero',  # Fix timestamp issues
                '-y',
                str(summary_path)
            ])
        
        try:
            logger.info(f"Running FFmpeg command: {' '.join(cmd[:10])}... (truncated)")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode != 0:
                logger.error(f"FFmpeg stderr: {result.stderr}")
                logger.error(f"FFmpeg stdout: {result.stdout}")
                raise RuntimeError(f"FFmpeg failed with code {result.returncode}: {result.stderr}")
            
            if summary_path.exists():
                logger.info(f"Video summary created successfully: {summary_path} ({summary_path.stat().st_size} bytes)")
            else:
                logger.error(f"Summary file was not created: {summary_path}")
                raise RuntimeError("Summary file was not created by FFmpeg")
                
            return summary_path
            
        except subprocess.TimeoutExpired:
            logger.error("Video summary creation timed out")
            raise RuntimeError("Video summary creation timed out")
        except FileNotFoundError:
            logger.error("FFmpeg not found - please ensure FFmpeg is installed")
            raise RuntimeError("FFmpeg not found - please ensure FFmpeg is installed")
        except Exception as e:
            logger.error(f"Video summary creation failed: {e}")
            raise RuntimeError(f"Video summary creation failed: {e}")
    
    def _create_vlc_bookmarks(self, video_path: Path, selected_scenes: List[Dict],
                             narrative_analysis: Dict) -> Path:
        """Create VLC bookmark file for the original video."""
        # Use custom output directory if set in session state
        import streamlit as st
        if hasattr(st, 'session_state') and hasattr(st.session_state, 'custom_output_dir') and st.session_state.custom_output_dir:
            base_output_dir = Path(st.session_state.custom_output_dir)
        else:
            base_output_dir = config.OUTPUT_DIR
            
        output_dir = base_output_dir / "bookmarks"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create bookmark with same name as video for VLC auto-detection
        bookmark_path = output_dir / f"{video_path.stem}.xspf"
        
        logger.info(f"Creating VLC bookmarks: {bookmark_path.name}")
        
        # Create XSPF playlist with bookmarks
        xspf_content = self._generate_xspf_content(video_path, selected_scenes, narrative_analysis)
        
        with open(bookmark_path, 'w', encoding='utf-8') as f:
            f.write(xspf_content)
        
        # Also create a simple instruction file  
        instruction_path = output_dir / f"{video_path.stem}_INSTRUCTIONS.txt"
        instructions = f"""VLC BOOKMARK INSTRUCTIONS for {video_path.name}
{'='*60}

HOW TO USE THESE BOOKMARKS:

1. OPEN IN VLC:
   - Double-click: {bookmark_path.name}
   - OR drag & drop this file into VLC
   - OR VLC Menu: Media → Open File → Select this .xspf file

2. VIEW BOOKMARKS:
   - In VLC: View menu → Playlist (Ctrl+L)
   - You'll see {len(selected_scenes)} key scenes listed

3. NAVIGATE:
   - Click any scene in the playlist to jump to that moment
   - Use Next/Previous buttons to move between key scenes
   - Each scene shows start time and description

SCENES INCLUDED:
"""
        
        for i, scene in enumerate(selected_scenes, 1):
            start_min = int(scene['start_time'] // 60)
            start_sec = int(scene['start_time'] % 60)
            duration_min = int(scene['duration'] // 60)
            duration_sec = int(scene['duration'] % 60)
            
            description = self._get_scene_description(scene, narrative_analysis)
            instructions += f"\n{i:2d}. {start_min:02d}:{start_sec:02d} - {description} ({duration_min:02d}:{duration_sec:02d})"
        
        instructions += f"\n\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        instructions += f"\nOriginal Video: {video_path}"
        
        with open(instruction_path, 'w', encoding='utf-8') as f:
            f.write(instructions)
        
        logger.info(f"VLC bookmarks created: {bookmark_path}")
        logger.info(f"Instructions created: {instruction_path}")
        return bookmark_path
    
    def _generate_xspf_content(self, video_path: Path, selected_scenes: List[Dict],
                              narrative_analysis: Dict) -> str:
        """Generate XSPF playlist content with bookmarks."""
        video_uri = video_path.as_uri()
        
        xspf_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<playlist xmlns="http://xspf.org/ns/0/" version="1">',
            f'    <title>Movie Summary - {video_path.stem}</title>',
            '    <trackList>'
        ]
        
        for scene in selected_scenes:
            start_time_ms = int(scene['start_time'] * 1000)
            duration_ms = int(scene['duration'] * 1000)
            
            # Create description from narrative analysis
            description = self._get_scene_description(scene, narrative_analysis)
            
            xspf_lines.extend([
                '        <track>',
                f'            <title>Scene {scene["scene_number"]}: {description}</title>',
                f'            <location>{video_uri}</location>',
                f'            <annotation>Start: {scene["start_time"]:.1f}s, Duration: {scene["duration"]:.1f}s</annotation>',
                '            <extension application="http://www.videolan.org/vlc/playlist/0">',
                f'                <vlc:option>start-time={start_time_ms}</vlc:option>',
                f'                <vlc:option>stop-time={start_time_ms + duration_ms}</vlc:option>',
                '            </extension>',
                '        </track>'
            ])
        
        xspf_lines.extend([
            '    </trackList>',
            '</playlist>'
        ])
        
        return '\n'.join(xspf_lines)
    
    def _get_scene_description(self, scene: Dict, narrative_analysis: Dict) -> str:
        """Get a description for the scene based on analysis."""
        # Try to match scene to key moments
        key_moments = narrative_analysis.get('key_moments', [])
        
        for moment in key_moments:
            if (moment.get('scene_number') == scene['scene_number'] or
                (moment.get('timestamp') and 
                 scene['start_time'] <= moment['timestamp'] <= scene['end_time'])):
                return moment.get('description', '').split('.')[0]  # First sentence
        
        # Default description based on importance
        importance = scene.get('combined_importance', 0.5)
        if importance > 0.8:
            return "Key narrative moment"
        elif importance > 0.6:
            return "Important scene"
        else:
            return "Supporting scene"
    
    def _create_scene_breakdown(self, selected_scenes: List[Dict], 
                               transcription_data: List[Dict],
                               narrative_analysis: Dict) -> List[Dict]:
        """Create detailed breakdown of each scene in the summary."""
        breakdown = []
        
        for scene in selected_scenes:
            # Get transcription for this scene
            scene_transcription = self._get_scene_transcription(scene, transcription_data)
            
            # Get narrative description
            description = self._get_scene_description(scene, narrative_analysis)
            
            scene_info = {
                'scene_number': scene['scene_number'],
                'start_time': scene['start_time'],
                'end_time': scene['end_time'],
                'duration': scene['duration'],
                'description': description,
                'transcription': scene_transcription,
                'importance_score': scene.get('combined_importance', 0.5),
                'frame_path': scene.get('frame_path')
            }
            
            breakdown.append(scene_info)
        
        return breakdown
    
    def _get_scene_transcription(self, scene: Dict, transcription_data: List[Dict]) -> str:
        """Get transcription text for a specific scene."""
        scene_start = scene['start_time']
        scene_end = scene['end_time']
        
        scene_text = []
        for trans in transcription_data:
            # Check if transcription overlaps with scene
            if not (trans['end_time'] < scene_start or trans['start_time'] > scene_end):
                scene_text.append(trans['text'].strip())
        
        return ' '.join(scene_text) if scene_text else "[No dialogue]"
    
    def _get_video_duration(self, video_path: Path) -> float:
        """Get video duration using ffprobe."""
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-show_entries', 'format=duration',
                '-of', 'csv=p=0',
                str(video_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                return float(result.stdout.strip())
            
        except (subprocess.TimeoutExpired, ValueError):
            pass
        
        # Fallback to OpenCV
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        
        return frame_count / fps if fps > 0 else 0
