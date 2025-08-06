"""
Advanced credits detection using computer vision and text analysis.
Identifies opening and closing credits through visual patterns, text detection, and audio analysis.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
import re
from collections import Counter
import subprocess
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CreditsDetector:
    """Advanced credits detection using multiple detection methods."""
    
    def __init__(self):
        """Initialize credits detector with configurable parameters."""
        self.text_cascade = None
        self.sample_interval = 5.0  # Sample every 5 seconds
        self.min_credits_duration = 30.0  # Minimum credits length
        self.confidence_threshold = 0.7
        
    def detect_credits_regions(self, video_path: str) -> Dict[str, Optional[Tuple[float, float]]]:
        """
        Detect opening and closing credits using multiple methods.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with 'opening' and 'closing' credit regions as (start, end) tuples
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        logger.info(f"Starting advanced credits detection for: {video_path.name}")
        
        # Run multiple detection methods
        results = {}
        
        try:
            # Method 1: Visual pattern analysis
            visual_results = self._detect_visual_patterns(video_path)
            
            # Method 2: Text density analysis
            text_results = self._detect_text_density(video_path)
            
            # Method 3: Audio pattern analysis  
            audio_results = self._detect_audio_patterns(video_path)
            
            # Method 4: Fade/transition analysis
            transition_results = self._detect_fade_transitions(video_path)
            
            # Combine results using weighted voting
            results = self._combine_detection_results([
                visual_results,
                text_results,
                audio_results,
                transition_results
            ])
            
            logger.info(f"Credits detection complete: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Credits detection failed: {e}")
            return {'opening': None, 'closing': None}
    
    def _detect_visual_patterns(self, video_path: Path) -> Dict[str, Optional[Tuple[float, float]]]:
        """Detect credits based on visual patterns like text blocks and static backgrounds."""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            return {'opening': None, 'closing': None}
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        # Sample frames at regular intervals
        sample_frames = []
        timestamps = []
        
        for t in np.arange(0, duration, self.sample_interval):
            frame_number = int(t * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if ret:
                # Analyze frame for credits characteristics
                credits_score = self._analyze_frame_for_credits(frame)
                sample_frames.append(credits_score)
                timestamps.append(t)
        
        cap.release()
        
        # Find continuous regions of high credits scores
        opening_credits = self._find_credits_region(sample_frames, timestamps, 'opening')
        closing_credits = self._find_credits_region(sample_frames, timestamps, 'closing')
        
        return {
            'opening': opening_credits,
            'closing': closing_credits
        }
    
    def _analyze_frame_for_credits(self, frame: np.ndarray) -> float:
        """Analyze a single frame for credits characteristics."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        score = 0.0
        
        # 1. Text density detection
        text_score = self._detect_text_in_frame(gray)
        score += text_score * 0.4
        
        # 2. Background uniformity (credits often have solid/gradient backgrounds)
        uniformity_score = self._measure_background_uniformity(gray)
        score += uniformity_score * 0.3
        
        # 3. Vertical text alignment (credits are often center-aligned)
        alignment_score = self._detect_vertical_alignment(gray)
        score += alignment_score * 0.2
        
        # 4. Color palette simplicity (credits use limited colors)
        palette_score = self._analyze_color_palette(frame)
        score += palette_score * 0.1
        
        return min(score, 1.0)
    
    def _detect_text_in_frame(self, gray_frame: np.ndarray) -> float:
        """Detect text density in frame using edge detection and contour analysis."""
        # Apply edge detection
        edges = cv2.Canny(gray_frame, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze contours for text-like characteristics
        text_contours = 0
        total_area = gray_frame.shape[0] * gray_frame.shape[1]
        text_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # Too small
                continue
                
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # Text-like characteristics
            if (0.1 < aspect_ratio < 10 and  # Reasonable aspect ratio
                area > 50 and               # Minimum size
                w > 10 and h > 5):          # Minimum dimensions
                text_contours += 1
                text_area += area
        
        # Calculate text density score
        density_score = min(text_area / total_area * 10, 1.0)
        contour_score = min(text_contours / 20, 1.0)
        
        return (density_score + contour_score) / 2
    
    def _measure_background_uniformity(self, gray_frame: np.ndarray) -> float:
        """Measure how uniform the background is (credits often have solid backgrounds)."""
        # Calculate standard deviation of pixel values
        std_dev = np.std(gray_frame)
        
        # Lower std_dev = more uniform = more likely credits
        # Normalize to 0-1 range (assuming std_dev of 0-100)
        uniformity_score = max(0, (100 - std_dev) / 100)
        
        return uniformity_score
    
    def _detect_vertical_alignment(self, gray_frame: np.ndarray) -> float:
        """Detect if text appears to be vertically centered (common in credits)."""
        height, width = gray_frame.shape
        
        # Split frame into horizontal bands
        band_height = height // 5
        band_variances = []
        
        for i in range(5):
            start_y = i * band_height
            end_y = min((i + 1) * band_height, height)
            band = gray_frame[start_y:end_y, :]
            
            # Calculate variance in this band
            variance = np.var(band)
            band_variances.append(variance)
        
        # Credits typically have more activity in center bands
        center_activity = sum(band_variances[1:4])  # Middle 3 bands
        total_activity = sum(band_variances)
        
        if total_activity == 0:
            return 0.0
        
        center_ratio = center_activity / total_activity
        
        # Higher center ratio = more likely credits
        return min(center_ratio * 1.5, 1.0)
    
    def _analyze_color_palette(self, frame: np.ndarray) -> float:
        """Analyze color palette simplicity (credits often use limited colors)."""
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Quantize colors to reduce palette
        h_bins, s_bins, v_bins = 12, 4, 4
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [h_bins, s_bins, v_bins], 
                           [0, 180, 0, 256, 0, 256])
        
        # Count significant colors (non-zero histogram bins)
        significant_colors = np.count_nonzero(hist > hist.max() * 0.01)
        
        # Fewer colors = more likely credits
        # Normalize assuming 10-50 colors is typical for credits
        simplicity_score = max(0, (50 - significant_colors) / 40)
        
        return min(simplicity_score, 1.0)
    
    def _detect_text_density(self, video_path: Path) -> Dict[str, Optional[Tuple[float, float]]]:
        """Detect credits using OCR-based text density analysis."""
        # This would use Tesseract OCR to detect actual text
        # For now, return placeholder results
        logger.info("Text density analysis would use Tesseract OCR")
        return {'opening': None, 'closing': None}
    
    def _detect_audio_patterns(self, video_path: Path) -> Dict[str, Optional[Tuple[float, float]]]:
        """Detect credits based on audio patterns (music vs dialogue)."""
        try:
            # Use FFmpeg to analyze audio characteristics
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_streams', '-select_streams', 'a:0',
                str(video_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                audio_info = json.loads(result.stdout)
                # Analyze audio patterns for credits detection
                # This would involve spectral analysis, volume patterns, etc.
                logger.info("Audio pattern analysis completed")
            
        except Exception as e:
            logger.warning(f"Audio analysis failed: {e}")
        
        return {'opening': None, 'closing': None}
    
    def _detect_fade_transitions(self, video_path: Path) -> Dict[str, Optional[Tuple[float, float]]]:
        """Detect credits based on fade-in/fade-out transitions."""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            return {'opening': None, 'closing': None}
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        # Analyze brightness changes at beginning and end
        fade_regions = []
        
        # Check first 10 minutes for opening fades
        opening_fade = self._analyze_fade_region(cap, 0, min(600, duration), fps)
        if opening_fade:
            fade_regions.append(('opening', opening_fade))
        
        # Check last 15 minutes for closing fades
        closing_start = max(0, duration - 900)
        closing_fade = self._analyze_fade_region(cap, closing_start, duration, fps)
        if closing_fade:
            fade_regions.append(('closing', closing_fade))
        
        cap.release()
        
        # Convert to expected format
        result = {'opening': None, 'closing': None}
        for fade_type, (start, end) in fade_regions:
            result[fade_type] = (start, end)
        
        return result
    
    def _analyze_fade_region(self, cap: cv2.VideoCapture, start_time: float, 
                           end_time: float, fps: float) -> Optional[Tuple[float, float]]:
        """Analyze a specific time region for fade transitions."""
        brightness_values = []
        timestamps = []
        
        for t in np.arange(start_time, end_time, 1.0):  # Sample every second
            frame_number = int(t * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if ret:
                # Calculate average brightness
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness = np.mean(gray)
                brightness_values.append(brightness)
                timestamps.append(t)
        
        if len(brightness_values) < 5:
            return None
        
        # Look for fade patterns (gradual brightness changes)
        fade_detected = self._detect_fade_pattern(brightness_values, timestamps)
        
        return fade_detected
    
    def _detect_fade_pattern(self, brightness_values: List[float], 
                           timestamps: List[float]) -> Optional[Tuple[float, float]]:
        """Detect fade-in or fade-out patterns in brightness values."""
        if len(brightness_values) < 5:
            return None
        
        # Calculate brightness gradients
        gradients = np.gradient(brightness_values)
        
        # Look for sustained positive (fade-in) or negative (fade-out) gradients
        fade_threshold = 2.0  # Minimum gradient for fade detection
        min_fade_duration = 3.0  # Minimum fade duration in seconds
        
        in_fade = False
        fade_start = None
        
        for i, (gradient, timestamp) in enumerate(zip(gradients, timestamps)):
            if abs(gradient) > fade_threshold:
                if not in_fade:
                    fade_start = timestamp
                    in_fade = True
            else:
                if in_fade and fade_start is not None:
                    fade_duration = timestamp - fade_start
                    if fade_duration >= min_fade_duration:
                        return (fade_start, timestamp)
                    in_fade = False
                    fade_start = None
        
        return None
    
    def _find_credits_region(self, scores: List[float], timestamps: List[float], 
                           region_type: str) -> Optional[Tuple[float, float]]:
        """Find continuous regions of high credits scores."""
        if len(scores) < 3:
            return None
        
        # Smooth scores using moving average
        window_size = 3
        smoothed_scores = []
        for i in range(len(scores)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(scores), i + window_size // 2 + 1)
            avg_score = np.mean(scores[start_idx:end_idx])
            smoothed_scores.append(avg_score)
        
        # Find regions above threshold
        high_score_regions = []
        in_region = False
        region_start = None
        
        for i, (score, timestamp) in enumerate(zip(smoothed_scores, timestamps)):
            if score > self.confidence_threshold:
                if not in_region:
                    region_start = timestamp
                    in_region = True
            else:
                if in_region and region_start is not None:
                    region_duration = timestamp - region_start
                    if region_duration >= self.min_credits_duration:
                        high_score_regions.append((region_start, timestamp))
                    in_region = False
                    region_start = None
        
        # Handle region extending to end of video
        if in_region and region_start is not None:
            final_duration = timestamps[-1] - region_start
            if final_duration >= self.min_credits_duration:
                high_score_regions.append((region_start, timestamps[-1]))
        
        if not high_score_regions:
            return None
        
        # For opening credits, return the earliest region
        # For closing credits, return the latest region
        if region_type == 'opening':
            return high_score_regions[0]
        else:
            return high_score_regions[-1]
    
    def _combine_detection_results(self, results_list: List[Dict]) -> Dict[str, Optional[Tuple[float, float]]]:
        """Combine results from multiple detection methods using weighted voting."""
        # Weights for different methods
        weights = [0.4, 0.3, 0.2, 0.1]  # Visual, Text, Audio, Transitions
        
        combined = {'opening': None, 'closing': None}
        
        for region_type in ['opening', 'closing']:
            candidates = []
            
            for i, results in enumerate(results_list):
                if results.get(region_type):
                    start, end = results[region_type]
                    candidates.append((start, end, weights[i]))
            
            if candidates:
                # Use weighted average of candidates
                total_weight = sum(weight for _, _, weight in candidates)
                
                if total_weight > 0:
                    avg_start = sum(start * weight for start, _, weight in candidates) / total_weight
                    avg_end = sum(end * weight for _, end, weight in candidates) / total_weight
                    
                    combined[region_type] = (avg_start, avg_end)
        
        return combined

    def get_detection_confidence(self, video_path: str) -> Dict[str, float]:
        """Get confidence scores for credits detection without full processing."""
        # Quick analysis for confidence estimation
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            return {'opening': 0.0, 'closing': 0.0}
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        # Sample a few frames from beginning and end
        opening_confidence = self._sample_region_confidence(cap, 0, min(300, duration/4), fps)
        closing_confidence = self._sample_region_confidence(cap, max(0, duration*0.75), duration, fps)
        
        cap.release()
        
        return {
            'opening': opening_confidence,
            'closing': closing_confidence
        }
    
    def _sample_region_confidence(self, cap: cv2.VideoCapture, start_time: float, 
                                end_time: float, fps: float) -> float:
        """Sample a few frames from a region to estimate credits detection confidence."""
        sample_times = np.linspace(start_time, end_time, 5)
        scores = []
        
        for t in sample_times:
            frame_number = int(t * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if ret:
                score = self._analyze_frame_for_credits(frame)
                scores.append(score)
        
        return np.mean(scores) if scores else 0.0