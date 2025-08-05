"""
Video discovery and metadata extraction module.
Scans directories for video files and extracts metadata using OpenCV and ffprobe.
"""
import os
import cv2
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Generator, Optional
import logging
from datetime import datetime

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoDiscovery:
    """Discovers and analyzes video files in specified directories."""
    
    def __init__(self, db=None):
        """Initialize video discovery engine."""
        self.supported_formats = config.SUPPORTED_FORMATS
        self.subtitle_formats = config.SUBTITLE_FORMATS
        self.db = db
    
    def scan_directory(self, root_path: str, include_subdirs: bool = True) -> Generator[Dict, None, None]:
        """
        Scan directory for video files and yield metadata.
        
        Args:
            root_path: Root directory to scan
            include_subdirs: Whether to scan subdirectories
            
        Yields:
            Dict containing video metadata
        """
        root_dir = Path(root_path)
        
        if not root_dir.exists():
            logger.error(f"Directory does not exist: {root_dir}")
            return
        
        logger.info(f"Scanning directory: {root_dir}")
        scan_start_time = datetime.now()
        video_count = 0
        total_size = 0
        
        # Find all video files
        pattern = "**/*" if include_subdirs else "*"
        
        for file_path in root_dir.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                try:
                    metadata = self.extract_metadata(file_path)
                    if metadata:
                        video_count += 1
                        total_size += metadata.get('file_size', 0)
                        yield metadata
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
        
        # Record this scan in recent folders if database is available
        if self.db:
            scan_duration = (datetime.now() - scan_start_time).total_seconds()
            description = f"Found {video_count} videos" if video_count > 0 else "No videos found"
            try:
                self.db.add_recent_folder(
                    folder_path=str(root_dir),
                    video_count=video_count,
                    total_size=total_size,
                    scan_duration=scan_duration,
                    description=description
                )
            except Exception as e:
                logger.warning(f"Could not save recent folder: {e}")
    
    def extract_metadata(self, file_path: Path) -> Optional[Dict]:
        """
        Extract comprehensive metadata from video file.
        
        Args:
            file_path: Path to video file
            
        Returns:
            Dictionary containing video metadata
        """
        try:
            # Basic file information
            stat = file_path.stat()
            metadata = {
                'file_path': str(file_path),
                'filename': file_path.name,
                'file_size': stat.st_size,
                'last_modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'extension': file_path.suffix.lower()
            }
            
            # Extract video metadata using OpenCV
            cv_metadata = self._extract_opencv_metadata(file_path)
            metadata.update(cv_metadata)
            
            # Try to get more detailed metadata using ffprobe
            ffprobe_metadata = self._extract_ffprobe_metadata(file_path)
            if ffprobe_metadata:
                metadata.update(ffprobe_metadata)
            
            # Check for subtitle files
            subtitle_info = self._find_subtitles(file_path)
            metadata.update(subtitle_info)
            
            logger.info(f"Extracted metadata for: {file_path.name}")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to extract metadata from {file_path}: {e}")
            return None
    
    def _extract_opencv_metadata(self, file_path: Path) -> Dict:
        """Extract basic metadata using OpenCV."""
        metadata = {}
        
        try:
            cap = cv2.VideoCapture(str(file_path))
            
            if cap.isOpened():
                # Basic video properties
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                metadata.update({
                    'fps': fps,
                    'frame_count': frame_count,
                    'duration': frame_count / fps if fps > 0 else 0,
                    'resolution': f"{width}x{height}",
                    'width': width,
                    'height': height
                })
                
                cap.release()
            
        except Exception as e:
            logger.warning(f"OpenCV metadata extraction failed for {file_path}: {e}")
        
        return metadata
    
    def _extract_ffprobe_metadata(self, file_path: Path) -> Optional[Dict]:
        """Extract detailed metadata using ffprobe."""
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                str(file_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                return self._parse_ffprobe_output(data)
            
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, json.JSONDecodeError) as e:
            logger.warning(f"ffprobe failed for {file_path}: {e}")
        except FileNotFoundError:
            logger.warning("ffprobe not found, using OpenCV metadata only")
        
        return None
    
    def _parse_ffprobe_output(self, data: Dict) -> Dict:
        """Parse ffprobe JSON output."""
        metadata = {}
        
        # Format information
        if 'format' in data:
            format_info = data['format']
            metadata.update({
                'duration': float(format_info.get('duration', 0)),
                'bitrate': int(format_info.get('bit_rate', 0)),
                'format_name': format_info.get('format_name'),
            })
        
        # Stream information
        if 'streams' in data:
            video_streams = [s for s in data['streams'] if s.get('codec_type') == 'video']
            audio_streams = [s for s in data['streams'] if s.get('codec_type') == 'audio']
            subtitle_streams = [s for s in data['streams'] if s.get('codec_type') == 'subtitle']
            
            if video_streams:
                video_stream = video_streams[0]
                metadata.update({
                    'video_codec': video_stream.get('codec_name'),
                    'fps': eval(video_stream.get('r_frame_rate', '0/1')),
                    'width': video_stream.get('width'),
                    'height': video_stream.get('height'),
                    'resolution': f"{video_stream.get('width')}x{video_stream.get('height')}"
                })
            
            if audio_streams:
                audio_stream = audio_streams[0]
                metadata.update({
                    'audio_codec': audio_stream.get('codec_name'),
                    'audio_channels': audio_stream.get('channels'),
                    'sample_rate': audio_stream.get('sample_rate')
                })
            
            metadata['has_embedded_subtitles'] = len(subtitle_streams) > 0
            metadata['subtitle_tracks'] = len(subtitle_streams)
        
        return metadata
    
    def _find_subtitles(self, video_path: Path) -> Dict:
        """Find external subtitle files for a video."""
        subtitle_info = {
            'has_subtitles': False,
            'subtitle_path': None,
            'subtitle_files': []
        }
        
        # Check for subtitle files with same name
        video_stem = video_path.stem
        video_dir = video_path.parent
        
        # Common subtitle naming patterns
        patterns = [
            f"{video_stem}.srt",
            f"{video_stem}.vtt",
            f"{video_stem}.ass",
            f"{video_stem}.ssa",
            f"{video_stem}.en.srt",
            f"{video_stem}.english.srt"
        ]
        
        found_subtitles = []
        for pattern in patterns:
            subtitle_path = video_dir / pattern
            if subtitle_path.exists():
                found_subtitles.append(str(subtitle_path))
        
        if found_subtitles:
            subtitle_info.update({
                'has_subtitles': True,
                'subtitle_path': found_subtitles[0],  # Use first found
                'subtitle_files': found_subtitles
            })
        
        return subtitle_info
    
    def get_directory_stats(self, root_path: str) -> Dict:
        """Get statistics about videos in a directory."""
        stats = {
            'total_videos': 0,
            'total_size_gb': 0.0,
            'total_duration_hours': 0.0,
            'formats': {},
            'resolutions': {},
            'with_subtitles': 0
        }
        
        for metadata in self.scan_directory(root_path):
            stats['total_videos'] += 1
            stats['total_size_gb'] += (metadata.get('file_size', 0) / 1024**3)
            stats['total_duration_hours'] += (metadata.get('duration', 0) / 3600)
            
            # Format distribution
            ext = metadata.get('extension', 'unknown')
            stats['formats'][ext] = stats['formats'].get(ext, 0) + 1
            
            # Resolution distribution
            res = metadata.get('resolution', 'unknown')
            stats['resolutions'][res] = stats['resolutions'].get(res, 0) + 1
            
            # Subtitle count
            if metadata.get('has_subtitles'):
                stats['with_subtitles'] += 1
        
        return stats
    
    def validate_video_file(self, file_path: Path) -> bool:
        """
        Validate if a video file can be processed.
        
        Args:
            file_path: Path to video file
            
        Returns:
            True if video can be processed
        """
        try:
            # Check file existence and size
            if not file_path.exists() or file_path.stat().st_size == 0:
                return False
            
            # Quick OpenCV validation
            cap = cv2.VideoCapture(str(file_path))
            is_valid = cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0
            cap.release()
            
            return is_valid
            
        except Exception:
            return False
