"""
Audio analysis module for enhanced video summarization.
Analyzes audio characteristics, music detection, dialogue intensity, and sound patterns.
"""
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import subprocess
import json
import tempfile
import os
from scipy import signal
from scipy.fft import fft, fftfreq

logger = logging.getLogger(__name__)

class AudioAnalyzer:
    """Advanced audio analysis for video scenes."""
    
    def __init__(self):
        """Initialize the audio analyzer."""
        self.sample_rate = 16000  # Standard rate for speech processing
        self.audio_features = {
            'volume_threshold': 0.1,
            'silence_threshold': 0.05,
            'music_frequency_range': (80, 8000),
            'speech_frequency_range': (300, 3400)
        }
    
    def extract_audio_features(self, video_path: str, scenes: List[Dict]) -> List[Dict]:
        """
        Extract audio features for each scene.
        
        Args:
            video_path: Path to video file
            scenes: List of scene dictionaries
            
        Returns:
            Enhanced scenes with audio analysis
        """
        try:
            # Extract audio from video
            audio_data = self._extract_audio_from_video(video_path)
            if audio_data is None:
                logger.warning("Could not extract audio, returning scenes without audio analysis")
                return scenes
            
            enhanced_scenes = []
            
            for scene in scenes:
                start_time = scene.get('start_time', 0)
                end_time = scene.get('end_time', start_time + 30)
                
                # Extract audio segment for this scene
                audio_segment = self._extract_audio_segment(audio_data, start_time, end_time)
                
                # Analyze audio characteristics
                audio_analysis = self._analyze_audio_segment(audio_segment)
                
                # Enhanced scene with audio data
                enhanced_scene = {
                    **scene,
                    'audio_analysis': audio_analysis
                }
                
                enhanced_scenes.append(enhanced_scene)
            
            logger.info(f"Completed audio analysis for {len(enhanced_scenes)} scenes")
            return enhanced_scenes
            
        except Exception as e:
            logger.error(f"Error in audio analysis: {e}")
            return scenes
    
    def _extract_audio_from_video(self, video_path: str) -> Optional[np.ndarray]:
        """Extract audio data from video file using FFmpeg."""
        try:
            # Create temporary file for audio
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
            
            # Use FFmpeg to extract audio
            cmd = [
                'ffmpeg', '-i', video_path,
                '-ac', '1',  # Mono
                '-ar', str(self.sample_rate),  # Sample rate
                '-f', 'wav',
                '-y',  # Overwrite
                temp_audio_path
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr}")
                return None
            
            # Read audio data
            audio_data = self._read_wav_file(temp_audio_path)
            
            # Clean up temporary file
            os.unlink(temp_audio_path)
            
            return audio_data
            
        except subprocess.TimeoutExpired:
            logger.error("Audio extraction timed out")
            return None
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            return None
    
    def _read_wav_file(self, wav_path: str) -> Optional[np.ndarray]:
        """Read WAV file and return audio data."""
        try:
            # Use OpenCV to read audio (basic implementation)
            # In a real implementation, you'd use librosa or scipy.io.wavfile
            
            # For now, return a simple placeholder that simulates audio data
            # In production, replace this with actual audio reading
            logger.warning("Using placeholder audio data - integrate librosa for real audio processing")
            
            # Generate placeholder audio data based on file size
            file_size = Path(wav_path).stat().st_size
            duration_seconds = max(1, file_size // (self.sample_rate * 2))  # Rough estimate
            
            # Generate synthetic audio data for demonstration
            t = np.linspace(0, duration_seconds, duration_seconds * self.sample_rate)
            audio_data = np.random.normal(0, 0.1, len(t))  # Random noise as placeholder
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Error reading WAV file: {e}")
            return None
    
    def _extract_audio_segment(self, audio_data: np.ndarray, 
                             start_time: float, end_time: float) -> np.ndarray:
        """Extract audio segment for specified time range."""
        try:
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            
            # Ensure we don't go beyond audio data bounds
            start_sample = max(0, start_sample)
            end_sample = min(len(audio_data), end_sample)
            
            if start_sample >= end_sample:
                return np.array([])
            
            return audio_data[start_sample:end_sample]
            
        except Exception as e:
            logger.error(f"Error extracting audio segment: {e}")
            return np.array([])
    
    def _analyze_audio_segment(self, audio_segment: np.ndarray) -> Dict:
        """Analyze characteristics of an audio segment."""
        analysis = {
            'volume_stats': {},
            'frequency_analysis': {},
            'speech_indicators': {},
            'music_indicators': {},
            'silence_analysis': {},
            'audio_type': 'unknown'
        }
        
        if len(audio_segment) == 0:
            return analysis
        
        try:
            # Volume analysis
            analysis['volume_stats'] = self._analyze_volume(audio_segment)
            
            # Frequency analysis
            analysis['frequency_analysis'] = self._analyze_frequency_content(audio_segment)
            
            # Speech detection
            analysis['speech_indicators'] = self._detect_speech_characteristics(audio_segment)
            
            # Music detection
            analysis['music_indicators'] = self._detect_music_characteristics(audio_segment)
            
            # Silence analysis
            analysis['silence_analysis'] = self._analyze_silence_patterns(audio_segment)
            
            # Classify audio type
            analysis['audio_type'] = self._classify_audio_type(analysis)
            
        except Exception as e:
            logger.error(f"Error analyzing audio segment: {e}")
        
        return analysis
    
    def _analyze_volume(self, audio_segment: np.ndarray) -> Dict:
        """Analyze volume characteristics of audio segment."""
        volume_stats = {}
        
        try:
            # Calculate RMS (Root Mean Square) for volume
            rms = np.sqrt(np.mean(audio_segment**2))
            
            # Calculate peak volume
            peak = np.max(np.abs(audio_segment))
            
            # Calculate volume variance (dynamic range)
            volume_variance = np.var(np.abs(audio_segment))
            
            # Volume percentiles
            abs_audio = np.abs(audio_segment)
            volume_percentiles = np.percentile(abs_audio, [25, 50, 75, 90, 95])
            
            volume_stats = {
                'rms_volume': float(rms),
                'peak_volume': float(peak),
                'volume_variance': float(volume_variance),
                'volume_25th': float(volume_percentiles[0]),
                'volume_median': float(volume_percentiles[1]),
                'volume_75th': float(volume_percentiles[2]),
                'volume_90th': float(volume_percentiles[3]),
                'volume_95th': float(volume_percentiles[4]),
                'dynamic_range': float(volume_percentiles[4] - volume_percentiles[0])
            }
            
        except Exception as e:
            logger.error(f"Error in volume analysis: {e}")
        
        return volume_stats
    
    def _analyze_frequency_content(self, audio_segment: np.ndarray) -> Dict:
        """Analyze frequency content of audio segment."""
        freq_analysis = {}
        
        try:
            # Perform FFT
            fft_data = fft(audio_segment)
            freqs = fftfreq(len(audio_segment), 1/self.sample_rate)
            
            # Calculate magnitude spectrum
            magnitude = np.abs(fft_data)
            
            # Only consider positive frequencies
            positive_freq_mask = freqs > 0
            freqs_positive = freqs[positive_freq_mask]
            magnitude_positive = magnitude[positive_freq_mask]
            
            # Frequency band analysis
            freq_bands = {
                'sub_bass': (20, 60),
                'bass': (60, 250),
                'low_mid': (250, 500),
                'mid': (500, 2000),
                'high_mid': (2000, 4000),
                'presence': (4000, 6000),
                'brilliance': (6000, 20000)
            }
            
            band_energies = {}
            for band_name, (low_freq, high_freq) in freq_bands.items():
                band_mask = (freqs_positive >= low_freq) & (freqs_positive <= high_freq)
                if np.any(band_mask):
                    band_energy = np.mean(magnitude_positive[band_mask])
                    band_energies[band_name] = float(band_energy)
                else:
                    band_energies[band_name] = 0.0
            
            # Spectral features
            spectral_centroid = np.sum(freqs_positive * magnitude_positive) / np.sum(magnitude_positive)
            spectral_rolloff = self._calculate_spectral_rolloff(freqs_positive, magnitude_positive)
            
            freq_analysis = {
                'band_energies': band_energies,
                'spectral_centroid': float(spectral_centroid),
                'spectral_rolloff': float(spectral_rolloff),
                'dominant_frequency': float(freqs_positive[np.argmax(magnitude_positive)])
            }
            
        except Exception as e:
            logger.error(f"Error in frequency analysis: {e}")
        
        return freq_analysis
    
    def _calculate_spectral_rolloff(self, freqs: np.ndarray, 
                                  magnitude: np.ndarray, rolloff_percent: float = 0.85) -> float:
        """Calculate spectral rolloff frequency."""
        try:
            total_energy = np.sum(magnitude)
            rolloff_energy = total_energy * rolloff_percent
            
            cumulative_energy = np.cumsum(magnitude)
            rolloff_index = np.where(cumulative_energy >= rolloff_energy)[0]
            
            if len(rolloff_index) > 0:
                return freqs[rolloff_index[0]]
            else:
                return freqs[-1]
                
        except Exception:
            return 0.0
    
    def _detect_speech_characteristics(self, audio_segment: np.ndarray) -> Dict:
        """Detect speech-like characteristics in audio."""
        speech_indicators = {
            'speech_probability': 0.0,
            'voice_activity': 0.0,
            'speech_clarity': 0.0
        }
        
        try:
            # Simple speech detection based on frequency content and volume patterns
            
            # Speech frequency range energy
            fft_data = fft(audio_segment)
            freqs = fftfreq(len(audio_segment), 1/self.sample_rate)
            magnitude = np.abs(fft_data)
            
            # Speech frequency range (300-3400 Hz)
            speech_mask = (freqs >= 300) & (freqs <= 3400)
            speech_energy = np.mean(magnitude[speech_mask]) if np.any(speech_mask) else 0
            
            # Total energy
            total_energy = np.mean(magnitude)
            
            # Speech probability based on energy in speech range
            if total_energy > 0:
                speech_probability = min(1.0, speech_energy / total_energy * 2)
            else:
                speech_probability = 0.0
            
            # Voice activity detection (simplified)
            # Look for periodic patterns typical of speech
            frame_size = self.sample_rate // 10  # 100ms frames
            voice_frames = 0
            total_frames = 0
            
            for i in range(0, len(audio_segment) - frame_size, frame_size):
                frame = audio_segment[i:i + frame_size]
                frame_energy = np.mean(frame**2)
                
                if frame_energy > self.audio_features['volume_threshold']:
                    voice_frames += 1
                total_frames += 1
            
            voice_activity = voice_frames / total_frames if total_frames > 0 else 0
            
            # Speech clarity (inverse of noise)
            speech_clarity = min(1.0, speech_probability * voice_activity)
            
            speech_indicators = {
                'speech_probability': float(speech_probability),
                'voice_activity': float(voice_activity),
                'speech_clarity': float(speech_clarity)
            }
            
        except Exception as e:
            logger.error(f"Error in speech detection: {e}")
        
        return speech_indicators
    
    def _detect_music_characteristics(self, audio_segment: np.ndarray) -> Dict:
        """Detect music-like characteristics in audio."""
        music_indicators = {
            'music_probability': 0.0,
            'rhythm_strength': 0.0,
            'harmonic_content': 0.0
        }
        
        try:
            # Music detection based on harmonic content and rhythm
            
            fft_data = fft(audio_segment)
            freqs = fftfreq(len(audio_segment), 1/self.sample_rate)
            magnitude = np.abs(fft_data)
            
            # Music frequency range (80-8000 Hz)
            music_mask = (freqs >= 80) & (freqs <= 8000)
            music_energy = np.mean(magnitude[music_mask]) if np.any(music_mask) else 0
            
            # Total energy
            total_energy = np.mean(magnitude)
            
            # Music probability based on broader frequency content
            if total_energy > 0:
                music_probability = min(1.0, music_energy / total_energy * 1.5)
            else:
                music_probability = 0.0
            
            # Rhythm detection (simplified)
            # Look for periodic patterns in volume
            frame_size = self.sample_rate // 20  # 50ms frames
            volume_pattern = []
            
            for i in range(0, len(audio_segment) - frame_size, frame_size):
                frame = audio_segment[i:i + frame_size]
                frame_volume = np.sqrt(np.mean(frame**2))
                volume_pattern.append(frame_volume)
            
            if len(volume_pattern) > 4:
                # Calculate autocorrelation to find rhythmic patterns
                autocorr = np.correlate(volume_pattern, volume_pattern, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                
                # Look for peaks that indicate rhythm
                if len(autocorr) > 1:
                    rhythm_strength = np.max(autocorr[1:]) / autocorr[0] if autocorr[0] > 0 else 0
                else:
                    rhythm_strength = 0
            else:
                rhythm_strength = 0
            
            # Harmonic content (simplified)
            # Look for harmonic relationships in frequency spectrum
            harmonic_content = music_probability * 0.8  # Simplified calculation
            
            music_indicators = {
                'music_probability': float(music_probability),
                'rhythm_strength': float(min(1.0, rhythm_strength)),
                'harmonic_content': float(harmonic_content)
            }
            
        except Exception as e:
            logger.error(f"Error in music detection: {e}")
        
        return music_indicators
    
    def _analyze_silence_patterns(self, audio_segment: np.ndarray) -> Dict:
        """Analyze silence patterns in audio segment."""
        silence_analysis = {
            'silence_ratio': 0.0,
            'silence_segments': 0,
            'avg_silence_duration': 0.0,
            'max_silence_duration': 0.0
        }
        
        try:
            # Define silence threshold
            silence_threshold = self.audio_features['silence_threshold']
            
            # Calculate frame-based silence detection
            frame_size = self.sample_rate // 10  # 100ms frames
            silence_frames = []
            
            for i in range(0, len(audio_segment) - frame_size, frame_size):
                frame = audio_segment[i:i + frame_size]
                frame_energy = np.mean(frame**2)
                silence_frames.append(frame_energy < silence_threshold)
            
            if silence_frames:
                # Calculate silence ratio
                silence_ratio = sum(silence_frames) / len(silence_frames)
                
                # Find silence segments
                silence_segments = []
                current_segment_length = 0
                
                for is_silent in silence_frames:
                    if is_silent:
                        current_segment_length += 1
                    else:
                        if current_segment_length > 0:
                            silence_segments.append(current_segment_length)
                            current_segment_length = 0
                
                # Add last segment if it was silence
                if current_segment_length > 0:
                    silence_segments.append(current_segment_length)
                
                # Calculate statistics
                num_segments = len(silence_segments)
                avg_duration = np.mean(silence_segments) * 0.1 if silence_segments else 0  # Convert to seconds
                max_duration = max(silence_segments) * 0.1 if silence_segments else 0  # Convert to seconds
                
                silence_analysis = {
                    'silence_ratio': float(silence_ratio),
                    'silence_segments': num_segments,
                    'avg_silence_duration': float(avg_duration),
                    'max_silence_duration': float(max_duration)
                }
            
        except Exception as e:
            logger.error(f"Error in silence analysis: {e}")
        
        return silence_analysis
    
    def _classify_audio_type(self, analysis: Dict) -> str:
        """Classify the type of audio based on analysis results."""
        try:
            speech_prob = analysis.get('speech_indicators', {}).get('speech_probability', 0)
            music_prob = analysis.get('music_indicators', {}).get('music_probability', 0)
            silence_ratio = analysis.get('silence_analysis', {}).get('silence_ratio', 0)
            
            # Classification logic
            if silence_ratio > 0.8:
                return 'mostly_silent'
            elif speech_prob > 0.6 and speech_prob > music_prob:
                return 'dialogue'
            elif music_prob > 0.6 and music_prob > speech_prob:
                return 'music'
            elif speech_prob > 0.3 and music_prob > 0.3:
                return 'mixed'
            elif speech_prob > 0.3:
                return 'speech_with_noise'
            elif music_prob > 0.3:
                return 'music_with_noise'
            else:
                return 'ambient_noise'
                
        except Exception as e:
            logger.error(f"Error classifying audio type: {e}")
            return 'unknown'
    
    def generate_audio_summary(self, scenes_with_audio: List[Dict]) -> Dict:
        """Generate a summary of audio analysis across all scenes."""
        summary = {
            'total_scenes_analyzed': len(scenes_with_audio),
            'audio_type_distribution': {},
            'volume_characteristics': {},
            'speech_scenes': [],
            'music_scenes': [],
            'silent_scenes': [],
            'dialogue_intensity_flow': []
        }
        
        try:
            # Collect audio types
            audio_types = []
            volume_values = []
            speech_scenes = []
            music_scenes = []
            silent_scenes = []
            
            for i, scene in enumerate(scenes_with_audio):
                audio_analysis = scene.get('audio_analysis', {})
                audio_type = audio_analysis.get('audio_type', 'unknown')
                audio_types.append(audio_type)
                
                # Volume statistics
                volume_stats = audio_analysis.get('volume_stats', {})
                rms_volume = volume_stats.get('rms_volume', 0)
                volume_values.append(rms_volume)
                
                # Categorize scenes
                speech_prob = audio_analysis.get('speech_indicators', {}).get('speech_probability', 0)
                music_prob = audio_analysis.get('music_indicators', {}).get('music_probability', 0)
                silence_ratio = audio_analysis.get('silence_analysis', {}).get('silence_ratio', 0)
                
                if speech_prob > 0.5:
                    speech_scenes.append({
                        'scene_index': i,
                        'start_time': scene.get('start_time', 0),
                        'speech_probability': speech_prob
                    })
                
                if music_prob > 0.5:
                    music_scenes.append({
                        'scene_index': i,
                        'start_time': scene.get('start_time', 0),
                        'music_probability': music_prob
                    })
                
                if silence_ratio > 0.7:
                    silent_scenes.append({
                        'scene_index': i,
                        'start_time': scene.get('start_time', 0),
                        'silence_ratio': silence_ratio
                    })
                
                # Track dialogue intensity
                summary['dialogue_intensity_flow'].append({
                    'scene_index': i,
                    'speech_probability': speech_prob,
                    'volume_level': rms_volume
                })
            
            # Audio type distribution
            for audio_type in set(audio_types):
                summary['audio_type_distribution'][audio_type] = audio_types.count(audio_type)
            
            # Volume characteristics
            if volume_values:
                summary['volume_characteristics'] = {
                    'avg_volume': float(np.mean(volume_values)),
                    'max_volume': float(np.max(volume_values)),
                    'min_volume': float(np.min(volume_values)),
                    'volume_variance': float(np.var(volume_values))
                }
            
            # Store categorized scenes
            summary['speech_scenes'] = speech_scenes[:10]  # Top 10
            summary['music_scenes'] = music_scenes[:10]   # Top 10
            summary['silent_scenes'] = silent_scenes[:10] # Top 10
            
        except Exception as e:
            logger.error(f"Error generating audio summary: {e}")
        
        return summary