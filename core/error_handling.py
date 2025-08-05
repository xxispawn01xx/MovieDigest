"""
Enhanced error handling for video processing operations.
Provides better logging, error recovery, and file validation.
"""
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class VideoProcessingErrorHandler:
    """Handles and categorizes video processing errors for better user feedback."""
    
    def __init__(self):
        """Initialize error handler with common error patterns."""
        self.ffmpeg_warning_patterns = [
            r"Unsupported encoding type",
            r"Header missing",
            r"invalid as first byte of an EBML number",
            r"corrupted data",
            r"truncated",
            r"non-monotonic DTS"
        ]
        
        self.critical_error_patterns = [
            r"No such file or directory",
            r"Permission denied",
            r"out of memory",
            r"disk full",
            r"cuda out of memory"
        ]
    
    def categorize_ffmpeg_output(self, output: str) -> Dict:
        """
        Categorize FFmpeg output into warnings and critical errors.
        
        Args:
            output: Raw FFmpeg output string
            
        Returns:
            Dictionary with categorized messages
        """
        warnings = []
        errors = []
        info_messages = []
        
        lines = output.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for critical errors
            is_critical = any(re.search(pattern, line, re.IGNORECASE) 
                            for pattern in self.critical_error_patterns)
            
            if is_critical:
                errors.append(line)
                continue
            
            # Check for warnings
            is_warning = any(re.search(pattern, line, re.IGNORECASE) 
                           for pattern in self.ffmpeg_warning_patterns)
            
            if is_warning:
                warnings.append(line)
            elif line.startswith('INFO:'):
                info_messages.append(line)
        
        return {
            'warnings': warnings,
            'errors': errors,
            'info': info_messages,
            'total_warnings': len(warnings),
            'total_errors': len(errors),
            'has_critical_errors': len(errors) > 0
        }
    
    def is_processable_video(self, file_path: Path) -> bool:
        """
        Check if a video file is likely processable despite warnings.
        
        Args:
            file_path: Path to video file
            
        Returns:
            True if video appears processable
        """
        try:
            # Basic file checks
            if not file_path.exists():
                return False
            
            if file_path.stat().st_size < 1024:  # Less than 1KB
                logger.warning(f"File too small: {file_path}")
                return False
            
            # Check file extension
            supported_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.wmv']
            if file_path.suffix.lower() not in supported_extensions:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking file {file_path}: {e}")
            return False
    
    def generate_processing_summary(self, processed_files: List[Dict]) -> str:
        """
        Generate a human-readable summary of processing results.
        
        Args:
            processed_files: List of processing results
            
        Returns:
            Formatted summary string
        """
        total_files = len(processed_files)
        successful = sum(1 for f in processed_files if f.get('success', False))
        failed = total_files - successful
        
        warnings_count = sum(f.get('warnings', 0) for f in processed_files)
        
        summary = f"""
Processing Summary:
  Total files: {total_files}
  Successful: {successful}
  Failed: {failed}
  Warnings: {warnings_count}
  
Success rate: {(successful/total_files*100):.1f}%
"""
        
        if failed > 0:
            summary += "\nFailed files:\n"
            for file_data in processed_files:
                if not file_data.get('success', False):
                    summary += f"  - {file_data.get('filename', 'Unknown')}: {file_data.get('error', 'Unknown error')}\n"
        
        return summary
    
    def should_continue_processing(self, error_rate: float, 
                                 consecutive_failures: int) -> bool:
        """
        Determine if batch processing should continue based on error patterns.
        
        Args:
            error_rate: Current error rate (0.0 to 1.0)
            consecutive_failures: Number of consecutive failures
            
        Returns:
            True if processing should continue
        """
        # Stop if error rate is too high
        if error_rate > 0.8:  # More than 80% failure rate
            logger.warning("High error rate detected, stopping processing")
            return False
        
        # Stop if too many consecutive failures
        if consecutive_failures > 5:
            logger.warning("Too many consecutive failures, stopping processing")
            return False
        
        return True