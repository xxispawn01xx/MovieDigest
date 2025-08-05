"""
VLC bookmark generation module for creating detailed bookmarks with key narrative moments.
Generates XSPF playlist files that can be loaded in VLC for easy navigation.
"""
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import logging
from urllib.parse import quote

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VLCBookmarkGenerator:
    """Generates VLC-compatible bookmark files for video navigation."""
    
    def __init__(self):
        """Initialize VLC bookmark generator."""
        self.output_dir = config.OUTPUT_DIR / "bookmarks"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_bookmarks(self, video_path: str, selected_scenes: List[Dict],
                        narrative_analysis: Dict, metadata: Dict = None) -> str:
        """
        Create VLC bookmark file with key narrative moments.
        
        Args:
            video_path: Path to the original video file
            selected_scenes: Scenes selected for the summary
            narrative_analysis: Narrative analysis results
            metadata: Optional video metadata
            
        Returns:
            Path to the created bookmark file
        """
        video_path = Path(video_path)
        
        logger.info(f"Creating VLC bookmarks for: {video_path.name}")
        
        try:
            # Generate bookmark data
            bookmark_data = self._prepare_bookmark_data(
                video_path, selected_scenes, narrative_analysis, metadata
            )
            
            # Create XSPF playlist
            xspf_content = self._generate_xspf_playlist(bookmark_data)
            
            # Save bookmark file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            bookmark_filename = f"{video_path.stem}_bookmarks_{timestamp}.xspf"
            bookmark_path = self.output_dir / bookmark_filename
            
            with open(bookmark_path, 'w', encoding='utf-8') as f:
                f.write(xspf_content)
            
            # Also create a simple text bookmark file for manual reference
            text_bookmark_path = self._create_text_bookmarks(bookmark_data, video_path)
            
            logger.info(f"VLC bookmarks created: {bookmark_path}")
            logger.info(f"Text bookmarks created: {text_bookmark_path}")
            
            return str(bookmark_path)
            
        except Exception as e:
            logger.error(f"Failed to create VLC bookmarks: {e}")
            raise
    
    def _prepare_bookmark_data(self, video_path: Path, selected_scenes: List[Dict],
                              narrative_analysis: Dict, metadata: Dict = None) -> Dict:
        """Prepare structured bookmark data."""
        
        # Get key moments from narrative analysis
        key_moments = narrative_analysis.get('key_moments', [])
        structure_analysis = narrative_analysis.get('structure_analysis', {})
        
        bookmarks = []
        
        # Add structure-based bookmarks (Act breaks)
        if structure_analysis:
            # Calculate approximate act boundaries based on total duration
            total_duration = selected_scenes[-1]['end_time'] if selected_scenes else 0
            
            if total_duration > 0:
                # Act 1 (Setup) - roughly first 25%
                act1_end = total_duration * 0.25
                bookmarks.append({
                    'timestamp': 0,
                    'title': 'Act 1: Setup',
                    'description': structure_analysis.get('act1', {}).get('description', 'Story setup and character introduction'),
                    'type': 'act_break',
                    'importance': 'high'
                })
                
                # Act 2 (Confrontation) - roughly 25% to 75%
                act2_start = total_duration * 0.25
                bookmarks.append({
                    'timestamp': act2_start,
                    'title': 'Act 2: Confrontation',
                    'description': structure_analysis.get('act2', {}).get('description', 'Main conflict and rising action'),
                    'type': 'act_break',
                    'importance': 'high'
                })
                
                # Act 3 (Resolution) - roughly final 25%
                act3_start = total_duration * 0.75
                bookmarks.append({
                    'timestamp': act3_start,
                    'title': 'Act 3: Resolution',
                    'description': structure_analysis.get('act3', {}).get('description', 'Climax and resolution'),
                    'type': 'act_break',
                    'importance': 'high'
                })
        
        # Add scene-based bookmarks
        for scene in selected_scenes:
            # Get scene description
            scene_description = self._get_scene_description(scene, narrative_analysis)
            
            # Determine bookmark importance
            importance = self._calculate_bookmark_importance(scene, key_moments)
            
            bookmark = {
                'timestamp': scene['start_time'],
                'title': f"Scene {scene['scene_number']}: {scene_description['title']}",
                'description': scene_description['description'],
                'duration': scene['duration'],
                'type': 'scene',
                'importance': importance,
                'scene_number': scene['scene_number'],
                'narrative_score': scene.get('narrative_importance', 0.5)
            }
            
            bookmarks.append(bookmark)
        
        # Add key moment bookmarks
        for moment in key_moments:
            if moment.get('timestamp'):
                bookmarks.append({
                    'timestamp': moment['timestamp'],
                    'title': f"Key Moment: {moment.get('description', '').split('.')[0]}",
                    'description': moment.get('description', 'Important narrative moment'),
                    'type': 'key_moment',
                    'importance': 'critical'
                })
        
        # Sort bookmarks by timestamp
        bookmarks.sort(key=lambda x: x['timestamp'])
        
        # Remove duplicates (bookmarks within 30 seconds of each other)
        filtered_bookmarks = self._remove_duplicate_bookmarks(bookmarks)
        
        bookmark_data = {
            'video_path': video_path,
            'video_title': video_path.stem.replace('_', ' ').title(),
            'total_duration': selected_scenes[-1]['end_time'] if selected_scenes else 0,
            'bookmarks': filtered_bookmarks,
            'metadata': metadata or {},
            'created_at': datetime.now().isoformat(),
            'summary_info': {
                'total_scenes': len(selected_scenes),
                'key_moments': len(key_moments),
                'narrative_summary': narrative_analysis.get('narrative_summary', '')
            }
        }
        
        return bookmark_data
    
    def _get_scene_description(self, scene: Dict, narrative_analysis: Dict) -> Dict:
        """Get descriptive title and description for a scene."""
        
        # Check if scene matches any key moments
        key_moments = narrative_analysis.get('key_moments', [])
        
        for moment in key_moments:
            if (moment.get('scene_number') == scene['scene_number'] or
                (moment.get('timestamp') and 
                 scene['start_time'] <= moment['timestamp'] <= scene['end_time'])):
                
                description_parts = moment.get('description', '').split('.')
                title = description_parts[0] if description_parts else 'Key Scene'
                description = moment.get('description', 'Important narrative moment')
                
                return {
                    'title': title,
                    'description': description
                }
        
        # Generate description based on scene properties
        importance = scene.get('combined_importance', 0.5)
        duration = scene['duration']
        
        if importance > 0.8:
            title = "Critical Scene"
            description = "High-importance scene crucial to the narrative"
        elif importance > 0.6:
            title = "Important Scene"
            description = "Significant scene advancing the plot"
        elif duration > 120:
            title = "Extended Scene"
            description = "Longer scene with detailed development"
        elif duration < 15:
            title = "Brief Moment"
            description = "Short but notable scene"
        else:
            title = "Supporting Scene"
            description = "Scene contributing to overall narrative"
        
        return {
            'title': title,
            'description': description
        }
    
    def _calculate_bookmark_importance(self, scene: Dict, key_moments: List[Dict]) -> str:
        """Calculate importance level for bookmark."""
        
        # Check if scene is a key moment
        for moment in key_moments:
            if (moment.get('scene_number') == scene['scene_number'] or
                (moment.get('timestamp') and 
                 scene['start_time'] <= moment['timestamp'] <= scene['end_time'])):
                return 'critical'
        
        # Base on combined importance score
        importance_score = scene.get('combined_importance', 0.5)
        
        if importance_score > 0.8:
            return 'high'
        elif importance_score > 0.6:
            return 'medium'
        else:
            return 'low'
    
    def _remove_duplicate_bookmarks(self, bookmarks: List[Dict], 
                                   time_threshold: float = 30.0) -> List[Dict]:
        """Remove bookmarks that are too close together in time."""
        
        if not bookmarks:
            return []
        
        filtered_bookmarks = [bookmarks[0]]
        
        for bookmark in bookmarks[1:]:
            # Check if bookmark is far enough from the last added bookmark
            last_timestamp = filtered_bookmarks[-1]['timestamp']
            current_timestamp = bookmark['timestamp']
            
            if current_timestamp - last_timestamp >= time_threshold:
                filtered_bookmarks.append(bookmark)
            else:
                # Keep the one with higher importance
                last_bookmark = filtered_bookmarks[-1]
                
                importance_order = {'critical': 3, 'high': 2, 'medium': 1, 'low': 0}
                
                current_importance = importance_order.get(bookmark['importance'], 0)
                last_importance = importance_order.get(last_bookmark['importance'], 0)
                
                if current_importance > last_importance:
                    filtered_bookmarks[-1] = bookmark
        
        logger.info(f"Filtered {len(bookmarks)} bookmarks to {len(filtered_bookmarks)}")
        return filtered_bookmarks
    
    def _generate_xspf_playlist(self, bookmark_data: Dict) -> str:
        """Generate XSPF playlist content."""
        
        video_path = bookmark_data['video_path']
        video_title = bookmark_data['video_title']
        bookmarks = bookmark_data['bookmarks']
        
        # Create root element
        playlist = ET.Element('playlist')
        playlist.set('xmlns', 'http://xspf.org/ns/0/')
        playlist.set('xmlns:vlc', 'http://www.videolan.org/vlc/playlist/ns/0/')
        playlist.set('version', '1')
        
        # Add playlist metadata
        title_elem = ET.SubElement(playlist, 'title')
        title_elem.text = f"{video_title} - Summary Bookmarks"
        
        creator_elem = ET.SubElement(playlist, 'creator')
        creator_elem.text = "Video Summarization Engine"
        
        info_elem = ET.SubElement(playlist, 'info')
        info_elem.text = f"Generated on {bookmark_data['created_at']}"
        
        # Create track list
        tracklist = ET.SubElement(playlist, 'trackList')
        
        # Add tracks for each bookmark
        for i, bookmark in enumerate(bookmarks):
            track = ET.SubElement(tracklist, 'track')
            
            # Track title
            track_title = ET.SubElement(track, 'title')
            track_title.text = bookmark['title']
            
            # Track location (video file)
            location = ET.SubElement(track, 'location')
            location.text = video_path.as_uri()
            
            # Track annotation
            annotation = ET.SubElement(track, 'annotation')
            annotation_text = f"{bookmark['description']}\n"
            annotation_text += f"Time: {self._format_time(bookmark['timestamp'])}"
            
            if 'duration' in bookmark:
                annotation_text += f" | Duration: {self._format_time(bookmark['duration'])}"
            
            annotation_text += f" | Importance: {bookmark['importance'].title()}"
            annotation.text = annotation_text
            
            # VLC-specific options
            extension = ET.SubElement(track, 'extension')
            extension.set('application', 'http://www.videolan.org/vlc/playlist/0')
            
            # Start time option
            start_time_ms = int(bookmark['timestamp'] * 1000)
            start_option = ET.SubElement(extension, 'vlc:option')
            start_option.text = f"start-time={start_time_ms}"
            
            # Stop time option (if duration is specified)
            if 'duration' in bookmark:
                stop_time_ms = int((bookmark['timestamp'] + bookmark['duration']) * 1000)
                stop_option = ET.SubElement(extension, 'vlc:option')
                stop_option.text = f"stop-time={stop_time_ms}"
            
            # Track ID
            track_id = ET.SubElement(extension, 'vlc:id')
            track_id.text = str(i)
        
        # Convert to string with proper formatting
        ET.indent(playlist, space="    ")
        
        # Create XML declaration and return
        xml_declaration = '<?xml version="1.0" encoding="UTF-8"?>\n'
        return xml_declaration + ET.tostring(playlist, encoding='unicode')
    
    def _create_text_bookmarks(self, bookmark_data: Dict, video_path: Path) -> str:
        """Create a simple text file with bookmark information."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        text_filename = f"{video_path.stem}_bookmarks_{timestamp}.txt"
        text_path = self.output_dir / text_filename
        
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(f"VIDEO SUMMARY BOOKMARKS\n")
            f.write(f"{'=' * 50}\n\n")
            f.write(f"Video: {bookmark_data['video_title']}\n")
            f.write(f"Total Duration: {self._format_time(bookmark_data['total_duration'])}\n")
            f.write(f"Created: {bookmark_data['created_at']}\n\n")
            
            # Summary information
            summary_info = bookmark_data['summary_info']
            f.write(f"SUMMARY INFORMATION\n")
            f.write(f"{'-' * 30}\n")
            f.write(f"Total Scenes: {summary_info['total_scenes']}\n")
            f.write(f"Key Moments: {summary_info['key_moments']}\n\n")
            
            if summary_info.get('narrative_summary'):
                f.write(f"NARRATIVE SUMMARY\n")
                f.write(f"{'-' * 30}\n")
                f.write(f"{summary_info['narrative_summary']}\n\n")
            
            # Bookmarks
            f.write(f"BOOKMARKS\n")
            f.write(f"{'-' * 30}\n\n")
            
            current_type = None
            for bookmark in bookmark_data['bookmarks']:
                # Group by type
                if bookmark['type'] != current_type:
                    current_type = bookmark['type']
                    f.write(f"\n{current_type.replace('_', ' ').title()}:\n")
                
                time_str = self._format_time(bookmark['timestamp'])
                importance_indicator = self._get_importance_indicator(bookmark['importance'])
                
                f.write(f"  {importance_indicator} {time_str} - {bookmark['title']}\n")
                f.write(f"    {bookmark['description']}\n")
                
                if 'duration' in bookmark:
                    duration_str = self._format_time(bookmark['duration'])
                    f.write(f"    Duration: {duration_str}\n")
                
                f.write(f"\n")
            
            # Instructions
            f.write(f"\nHOW TO USE\n")
            f.write(f"{'-' * 30}\n")
            f.write(f"1. Open VLC Media Player\n")
            f.write(f"2. Go to Media > Open Playlist\n")
            f.write(f"3. Select the .xspf file with the same name\n")
            f.write(f"4. Click on any bookmark to jump to that moment\n")
            f.write(f"5. Use the timestamps above to manually seek in any player\n")
        
        return str(text_path)
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
    
    def _get_importance_indicator(self, importance: str) -> str:
        """Get visual indicator for importance level."""
        indicators = {
            'critical': '🔴',
            'high': '🟡',
            'medium': '🟢',
            'low': '⚪'
        }
        return indicators.get(importance, '⚪')
    
    def create_chapter_file(self, video_path: str, selected_scenes: List[Dict]) -> str:
        """
        Create a chapter file for video players that support chapters.
        
        Args:
            video_path: Path to video file
            selected_scenes: Selected scenes for chapters
            
        Returns:
            Path to created chapter file
        """
        video_path = Path(video_path)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chapter_filename = f"{video_path.stem}_chapters_{timestamp}.txt"
        chapter_path = self.output_dir / chapter_filename
        
        with open(chapter_path, 'w', encoding='utf-8') as f:
            for i, scene in enumerate(selected_scenes):
                start_time = self._format_time_milliseconds(scene['start_time'])
                end_time = self._format_time_milliseconds(scene['end_time'])
                
                f.write(f"CHAPTER{i+1:02d}={start_time}\n")
                f.write(f"CHAPTER{i+1:02d}NAME=Scene {scene['scene_number']}\n")
        
        logger.info(f"Chapter file created: {chapter_path}")
        return str(chapter_path)
    
    def _format_time_milliseconds(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS.mmm for chapter files."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"
    
    def validate_bookmark_file(self, bookmark_path: str) -> bool:
        """
        Validate that a bookmark file is properly formatted.
        
        Args:
            bookmark_path: Path to bookmark file
            
        Returns:
            True if valid
        """
        try:
            with open(bookmark_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try to parse as XML
            ET.fromstring(content)
            
            # Check for required elements
            if 'trackList' in content and 'vlc:option' in content:
                logger.info(f"Bookmark file validation successful: {bookmark_path}")
                return True
            else:
                logger.warning(f"Bookmark file missing required elements: {bookmark_path}")
                return False
                
        except Exception as e:
            logger.error(f"Bookmark file validation failed: {e}")
            return False
