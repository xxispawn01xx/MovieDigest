"""
VLC bookmark generator for creating XSPF playlists with key scene timestamps.
Integrates with narrative analysis to create meaningful navigation points.
"""
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class VLCBookmarkGenerator:
    """Generates VLC-compatible bookmark files for video navigation."""
    
    def __init__(self, output_dir: str = "output/bookmarks"):
        """Initialize bookmark generator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_bookmark_playlist(self, video_data: Dict, output_path: Optional[str] = None) -> str:
        """
        Create VLC bookmark playlist from video analysis data.
        
        Args:
            video_data: Complete video analysis data including key moments
            output_path: Optional custom output path
            
        Returns:
            Path to created XSPF playlist file
        """
        try:
            if not output_path:
                video_name = Path(video_data.get('file_path', 'unknown')).stem
                output_path = self.output_dir / f"{video_name}_bookmarks.xspf"
            
            # Create XSPF playlist structure
            root = ET.Element('playlist')
            root.set('version', '1')
            root.set('xmlns', 'http://xspf.org/ns/0/')
            
            # Add playlist metadata
            title_elem = ET.SubElement(root, 'title')
            title_elem.text = f"Key Moments - {video_data.get('title', 'Unknown Video')}"
            
            creator_elem = ET.SubElement(root, 'creator')
            creator_elem.text = "Video Summarization Engine"
            
            # Add track list
            tracklist = ET.SubElement(root, 'trackList')
            
            # Get key moments from narrative analysis
            narrative_analysis = video_data.get('narrative_analysis', {})
            key_moments = narrative_analysis.get('key_moments', [])
            
            video_path = video_data.get('file_path', '')
            video_title = video_data.get('title', Path(video_path).stem)
            
            # Sort key moments by timestamp
            sorted_moments = sorted(key_moments, key=lambda x: x.get('timestamp', 0))
            
            for i, moment in enumerate(sorted_moments):
                track = ET.SubElement(tracklist, 'track')
                
                # Track location (file URI)
                location = ET.SubElement(track, 'location')
                location.text = f"file://{Path(video_path).absolute()}"
                
                # Track title with timestamp
                track_title = ET.SubElement(track, 'title')
                timestamp = moment.get('timestamp', 0)
                minutes, seconds = divmod(int(timestamp), 60)
                hours, minutes = divmod(minutes, 60)
                
                if hours > 0:
                    time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                else:
                    time_str = f"{minutes:02d}:{seconds:02d}"
                
                description = moment.get('description', f'Key Moment {i+1}')
                track_title.text = f"[{time_str}] {description}"
                
                # Add creator info
                track_creator = ET.SubElement(track, 'creator')
                track_creator.text = video_title
                
                # Add duration (estimate 30 seconds per bookmark)
                duration = ET.SubElement(track, 'duration')
                duration.text = "30000"  # 30 seconds in milliseconds
                
                # VLC extension for start time
                extension = ET.SubElement(track, 'extension')
                extension.set('application', 'http://www.videolan.org/vlc/playlist/ns/0/')
                
                # VLC option for start time
                vlc_option = ET.SubElement(extension, 'vlc:option')
                vlc_option.text = f"start-time={int(timestamp)}"
                
                # VLC option for stop time (30 seconds later)
                vlc_stop_option = ET.SubElement(extension, 'vlc:option')
                vlc_stop_option.text = f"stop-time={int(timestamp) + 30}"
            
            # Write XSPF file with proper formatting
            self._write_formatted_xml(root, output_path)
            
            logger.info(f"VLC bookmark playlist created: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to create VLC bookmark playlist: {e}")
            raise
    
    def create_chapter_playlist(self, video_data: Dict, output_path: Optional[str] = None) -> str:
        """
        Create chapter-based playlist for act structure navigation.
        
        Args:
            video_data: Video analysis data with structure information
            output_path: Optional custom output path
            
        Returns:
            Path to created chapter playlist
        """
        try:
            if not output_path:
                video_name = Path(video_data.get('file_path', 'unknown')).stem
                output_path = self.output_dir / f"{video_name}_chapters.xspf"
            
            root = ET.Element('playlist')
            root.set('version', '1')
            root.set('xmlns', 'http://xspf.org/ns/0/')
            
            # Playlist metadata
            title_elem = ET.SubElement(root, 'title')
            title_elem.text = f"Chapters - {video_data.get('title', 'Unknown Video')}"
            
            tracklist = ET.SubElement(root, 'trackList')
            
            # Get scene data for chapter creation
            scenes = video_data.get('scenes', [])
            total_duration = video_data.get('duration', 0)
            video_path = video_data.get('file_path', '')
            
            if scenes and len(scenes) >= 5:
                # Create 5-act structure chapters
                act_boundaries = self._calculate_act_boundaries(len(scenes))
                act_names = ['Setup', 'Rising Action', 'Midpoint', 'Climax', 'Resolution']
                
                for i, (act_name, scene_idx) in enumerate(zip(act_names, act_boundaries)):
                    if scene_idx < len(scenes):
                        track = ET.SubElement(tracklist, 'track')
                        
                        # Track location
                        location = ET.SubElement(track, 'location')
                        location.text = f"file://{Path(video_path).absolute()}"
                        
                        # Track title
                        track_title = ET.SubElement(track, 'title')
                        start_time = scenes[scene_idx]['start_time']
                        minutes, seconds = divmod(int(start_time), 60)
                        track_title.text = f"Act {i+1}: {act_name} ({minutes:02d}:{seconds:02d})"
                        
                        # VLC extension
                        extension = ET.SubElement(track, 'extension')
                        extension.set('application', 'http://www.videolan.org/vlc/playlist/ns/0/')
                        
                        vlc_option = ET.SubElement(extension, 'vlc:option')
                        vlc_option.text = f"start-time={int(start_time)}"
            
            self._write_formatted_xml(root, output_path)
            
            logger.info(f"Chapter playlist created: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to create chapter playlist: {e}")
            raise
    
    def create_summary_playlist(self, video_data: Dict, summary_scenes: List[int], 
                              output_path: Optional[str] = None) -> str:
        """
        Create playlist containing only the most important scenes for summary viewing.
        
        Args:
            video_data: Complete video analysis data
            summary_scenes: List of scene indices to include in summary
            output_path: Optional custom output path
            
        Returns:
            Path to created summary playlist
        """
        try:
            if not output_path:
                video_name = Path(video_data.get('file_path', 'unknown')).stem
                output_path = self.output_dir / f"{video_name}_summary.xspf"
            
            root = ET.Element('playlist')
            root.set('version', '1')
            root.set('xmlns', 'http://xspf.org/ns/0/')
            
            # Playlist metadata
            title_elem = ET.SubElement(root, 'title')
            title_elem.text = f"Summary Scenes - {video_data.get('title', 'Unknown Video')}"
            
            tracklist = ET.SubElement(root, 'trackList')
            
            scenes = video_data.get('scenes', [])
            video_path = video_data.get('file_path', '')
            
            for i, scene_idx in enumerate(sorted(summary_scenes)):
                if scene_idx < len(scenes):
                    scene = scenes[scene_idx]
                    track = ET.SubElement(tracklist, 'track')
                    
                    # Track location
                    location = ET.SubElement(track, 'location')
                    location.text = f"file://{Path(video_path).absolute()}"
                    
                    # Track title
                    track_title = ET.SubElement(track, 'title')
                    start_time = scene['start_time']
                    end_time = scene['end_time']
                    duration = end_time - start_time
                    
                    minutes, seconds = divmod(int(start_time), 60)
                    track_title.text = f"Scene {scene_idx + 1} ({minutes:02d}:{seconds:02d}) - {duration:.1f}s"
                    
                    # VLC extension with start and stop times
                    extension = ET.SubElement(track, 'extension')
                    extension.set('application', 'http://www.videolan.org/vlc/playlist/ns/0/')
                    
                    vlc_start = ET.SubElement(extension, 'vlc:option')
                    vlc_start.text = f"start-time={int(start_time)}"
                    
                    vlc_stop = ET.SubElement(extension, 'vlc:option')
                    vlc_stop.text = f"stop-time={int(end_time)}"
            
            self._write_formatted_xml(root, output_path)
            
            logger.info(f"Summary playlist created: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to create summary playlist: {e}")
            raise
    
    def _calculate_act_boundaries(self, total_scenes: int) -> List[int]:
        """Calculate scene indices for 5-act structure boundaries."""
        return [
            0,  # Act 1: Setup (0%)
            int(total_scenes * 0.25),  # Act 2: Rising Action (25%)
            int(total_scenes * 0.5),   # Act 3: Midpoint (50%)
            int(total_scenes * 0.75),  # Act 4: Climax (75%)
            int(total_scenes * 0.9)    # Act 5: Resolution (90%)
        ]
    
    def _write_formatted_xml(self, root: ET.Element, output_path: Path):
        """Write XML with proper formatting and encoding."""
        # Create formatted XML string
        xml_str = ET.tostring(root, encoding='unicode')
        
        # Parse and reformat for readability
        dom = ET.ElementTree(root)
        
        # Write with declaration
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            dom.write(f, encoding='unicode', xml_declaration=False)
    
    def get_bookmark_info(self, playlist_path: str) -> Dict:
        """
        Get information about an existing bookmark playlist.
        
        Args:
            playlist_path: Path to XSPF playlist file
            
        Returns:
            Dictionary with playlist information
        """
        try:
            tree = ET.parse(playlist_path)
            root = tree.getroot()
            
            # Extract namespace
            ns = {'xspf': 'http://xspf.org/ns/0/'}
            
            title = root.find('xspf:title', ns)
            tracks = root.findall('.//xspf:track', ns)
            
            info = {
                'title': title.text if title is not None else 'Unknown',
                'track_count': len(tracks),
                'file_path': playlist_path,
                'created_date': datetime.fromtimestamp(Path(playlist_path).stat().st_mtime),
                'bookmarks': []
            }
            
            # Extract bookmark information
            for track in tracks:
                track_title = track.find('xspf:title', ns)
                location = track.find('xspf:location', ns)
                
                if track_title is not None:
                    info['bookmarks'].append({
                        'title': track_title.text,
                        'location': location.text if location is not None else None
                    })
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to read bookmark playlist info: {e}")
            return {}