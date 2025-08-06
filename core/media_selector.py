"""
Media track selection for audio and subtitle streams in video files.
Handles multiple audio tracks and subtitle options during processing.
"""
import json
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class MediaTrackSelector:
    """Manages audio and subtitle track selection for video processing."""
    
    def __init__(self):
        """Initialize media track selector."""
        pass
    
    def analyze_media_streams(self, video_path: str) -> Dict:
        """
        Analyze all audio and subtitle streams in a video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with stream information
        """
        try:
            # Use ffprobe to get detailed stream information
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_streams',
                str(video_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"ffprobe failed: {result.stderr}")
                return self._get_default_streams()
            
            data = json.loads(result.stdout)
            streams = data.get('streams', [])
            
            # Categorize streams
            video_streams = []
            audio_streams = []
            subtitle_streams = []
            
            for i, stream in enumerate(streams):
                codec_type = stream.get('codec_type', '').lower()
                
                if codec_type == 'video':
                    video_streams.append(self._parse_video_stream(stream, i))
                elif codec_type == 'audio':
                    audio_streams.append(self._parse_audio_stream(stream, i))
                elif codec_type == 'subtitle':
                    subtitle_streams.append(self._parse_subtitle_stream(stream, i))
            
            # Find external subtitle files
            external_subtitles = self._find_external_subtitles(video_path)
            
            stream_info = {
                'video_streams': video_streams,
                'audio_streams': audio_streams,
                'subtitle_streams': subtitle_streams,
                'external_subtitles': external_subtitles,
                'has_multiple_audio': len(audio_streams) > 1,
                'has_subtitles': len(subtitle_streams) > 0 or len(external_subtitles) > 0,
                'recommended_audio': self._recommend_audio_track(audio_streams),
                'recommended_subtitle': self._recommend_subtitle_track(subtitle_streams, external_subtitles)
            }
            
            logger.info(f"Found {len(audio_streams)} audio tracks, {len(subtitle_streams)} subtitle tracks")
            return stream_info
            
        except Exception as e:
            logger.error(f"Failed to analyze streams: {e}")
            return self._get_default_streams()
    
    def _parse_video_stream(self, stream: Dict, index: int) -> Dict:
        """Parse video stream information."""
        return {
            'index': index,
            'codec_name': stream.get('codec_name', 'unknown'),
            'width': stream.get('width', 0),
            'height': stream.get('height', 0),
            'duration': float(stream.get('duration', 0)),
            'frame_rate': stream.get('avg_frame_rate', '0/1')
        }
    
    def _parse_audio_stream(self, stream: Dict, index: int) -> Dict:
        """Parse audio stream information."""
        tags = stream.get('tags', {})
        
        return {
            'index': index,
            'stream_index': stream.get('index', index),
            'codec_name': stream.get('codec_name', 'unknown'),
            'channels': stream.get('channels', 2),
            'sample_rate': stream.get('sample_rate', '44100'),
            'language': tags.get('language', 'und'),
            'title': tags.get('title', ''),
            'duration': float(stream.get('duration', 0)),
            'is_default': stream.get('disposition', {}).get('default', 0) == 1,
            'description': self._describe_audio_track(stream, tags)
        }
    
    def _parse_subtitle_stream(self, stream: Dict, index: int) -> Dict:
        """Parse subtitle stream information."""
        tags = stream.get('tags', {})
        
        return {
            'index': index,
            'stream_index': stream.get('index', index),
            'codec_name': stream.get('codec_name', 'unknown'),
            'language': tags.get('language', 'und'),
            'title': tags.get('title', ''),
            'is_default': stream.get('disposition', {}).get('default', 0) == 1,
            'is_forced': stream.get('disposition', {}).get('forced', 0) == 1,
            'description': self._describe_subtitle_track(stream, tags)
        }
    
    def _describe_audio_track(self, stream: Dict, tags: Dict) -> str:
        """Create human-readable audio track description."""
        parts = []
        
        # Language
        language = tags.get('language', 'unknown')
        if language != 'und':
            parts.append(language.upper())
        
        # Title or description
        title = tags.get('title', '')
        if title:
            parts.append(title)
        
        # Channel info
        channels = stream.get('channels', 2)
        if channels == 1:
            parts.append('Mono')
        elif channels == 2:
            parts.append('Stereo')
        elif channels > 2:
            parts.append(f'{channels}-Channel')
        
        # Codec
        codec = stream.get('codec_name', 'unknown')
        if codec != 'unknown':
            parts.append(codec.upper())
        
        return ' • '.join(parts) if parts else f'Audio Track {stream.get("index", 0)}'
    
    def _describe_subtitle_track(self, stream: Dict, tags: Dict) -> str:
        """Create human-readable subtitle track description."""
        parts = []
        
        # Language
        language = tags.get('language', 'unknown')
        if language != 'und':
            parts.append(language.upper())
        
        # Title
        title = tags.get('title', '')
        if title:
            parts.append(title)
        
        # Forced/Default flags
        disposition = stream.get('disposition', {})
        if disposition.get('forced', 0):
            parts.append('Forced')
        if disposition.get('default', 0):
            parts.append('Default')
        
        # Codec
        codec = stream.get('codec_name', 'unknown')
        if codec != 'unknown':
            parts.append(codec.upper())
        
        return ' • '.join(parts) if parts else f'Subtitle Track {stream.get("index", 0)}'
    
    def _find_external_subtitles(self, video_path: str) -> List[Dict]:
        """Find external subtitle files matching the video."""
        video_path = Path(video_path)
        video_stem = video_path.stem
        video_dir = video_path.parent
        
        # Common subtitle extensions and language patterns
        patterns = [
            f"{video_stem}.srt",
            f"{video_stem}.vtt", 
            f"{video_stem}.ass",
            f"{video_stem}.ssa",
            f"{video_stem}.sub",
            f"{video_stem}.idx",
            # Language-specific patterns
            f"{video_stem}.en.srt",
            f"{video_stem}.eng.srt",
            f"{video_stem}.english.srt",
            f"{video_stem}.es.srt",
            f"{video_stem}.spa.srt",
            f"{video_stem}.spanish.srt",
            f"{video_stem}.fr.srt",
            f"{video_stem}.fra.srt",
            f"{video_stem}.french.srt",
        ]
        
        external_subs = []
        for i, pattern in enumerate(patterns):
            subtitle_path = video_dir / pattern
            if subtitle_path.exists():
                # Extract language from filename
                language = 'unknown'
                if '.en.' in pattern or '.eng.' in pattern or '.english.' in pattern:
                    language = 'en'
                elif '.es.' in pattern or '.spa.' in pattern or '.spanish.' in pattern:
                    language = 'es' 
                elif '.fr.' in pattern or '.fra.' in pattern or '.french.' in pattern:
                    language = 'fr'
                
                external_subs.append({
                    'index': i,
                    'path': str(subtitle_path),
                    'filename': subtitle_path.name,
                    'language': language,
                    'format': subtitle_path.suffix.lower()[1:],  # Remove dot
                    'description': f"{language.upper()} • {subtitle_path.suffix.upper()[1:]} • External"
                })
        
        return external_subs
    
    def _recommend_audio_track(self, audio_streams: List[Dict]) -> Optional[int]:
        """Recommend the best audio track to use."""
        if not audio_streams:
            return None
        
        # Priority: Default track > English > Most channels > First track
        for stream in audio_streams:
            if stream.get('is_default', False):
                return stream['index']
        
        # Look for English
        for stream in audio_streams:
            if stream.get('language', '').lower() in ['en', 'eng', 'english']:
                return stream['index']
        
        # Most channels (likely main audio)
        best_stream = max(audio_streams, key=lambda s: s.get('channels', 0))
        return best_stream['index']
    
    def _recommend_subtitle_track(self, subtitle_streams: List[Dict], 
                                external_subtitles: List[Dict]) -> Optional[Dict]:
        """Recommend the best subtitle track."""
        all_subtitles = subtitle_streams + external_subtitles
        
        if not all_subtitles:
            return None
        
        # Priority: Default > English > First available
        for sub in all_subtitles:
            if sub.get('is_default', False):
                return sub
        
        # Look for English
        for sub in all_subtitles:
            if sub.get('language', '').lower() in ['en', 'eng', 'english']:
                return sub
        
        # Return first available
        return all_subtitles[0] if all_subtitles else None
    
    def _get_default_streams(self) -> Dict:
        """Return default stream information when analysis fails."""
        return {
            'video_streams': [],
            'audio_streams': [{'index': 0, 'description': 'Default Audio', 'language': 'unknown'}],
            'subtitle_streams': [],
            'external_subtitles': [],
            'has_multiple_audio': False,
            'has_subtitles': False,
            'recommended_audio': 0,
            'recommended_subtitle': None
        }
    
    def build_ffmpeg_audio_args(self, selected_audio_index: int = None) -> List[str]:
        """Build FFmpeg arguments for audio track selection."""
        args = []
        
        if selected_audio_index is not None:
            args.extend(['-map', f'0:a:{selected_audio_index}'])
        else:
            args.extend(['-map', '0:a:0'])  # Default to first audio track
        
        return args
    
    def build_ffmpeg_subtitle_args(self, subtitle_info: Dict = None) -> List[str]:
        """Build FFmpeg arguments for subtitle handling."""
        args = []
        
        if not subtitle_info:
            return args
        
        if 'path' in subtitle_info:
            # External subtitle file
            args.extend(['-i', subtitle_info['path']])
            args.extend(['-map', '1:s:0'])  # Map the subtitle from second input
            args.extend(['-c:s', 'srt'])  # Convert to SRT format
        elif 'stream_index' in subtitle_info:
            # Internal subtitle stream
            args.extend(['-map', f'0:s:{subtitle_info["stream_index"]}'])
            args.extend(['-c:s', 'srt'])
        
        return args
    
    def get_stream_info_for_ui(self, video_path: str) -> Dict:
        """Get formatted stream information for user interface display."""
        stream_info = self.analyze_media_streams(video_path)
        
        # Format for UI consumption
        ui_info = {
            'audio_options': [],
            'subtitle_options': [{'index': -1, 'description': 'No Subtitles', 'value': None}],
            'recommended_audio': stream_info.get('recommended_audio', 0),
            'recommended_subtitle': stream_info.get('recommended_subtitle'),
            'has_multiple_options': stream_info.get('has_multiple_audio', False) or stream_info.get('has_subtitles', False)
        }
        
        # Add audio options
        for stream in stream_info.get('audio_streams', []):
            ui_info['audio_options'].append({
                'index': stream['index'],
                'description': stream['description'],
                'value': stream['index']
            })
        
        # Add internal subtitle options
        for stream in stream_info.get('subtitle_streams', []):
            ui_info['subtitle_options'].append({
                'index': stream['index'],
                'description': stream['description'],
                'value': {'type': 'internal', 'stream_index': stream['stream_index']}
            })
        
        # Add external subtitle options
        for sub in stream_info.get('external_subtitles', []):
            ui_info['subtitle_options'].append({
                'index': sub['index'] + 1000,  # Offset to avoid conflicts
                'description': sub['description'],
                'value': {'type': 'external', 'path': sub['path']}
            })
        
        return ui_info