"""
Offline transcription module adapted from coursequery.
Uses Whisper for GPU-accelerated speech-to-text conversion.
"""
import torch
import whisper
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Optional
import logging
import json
import warnings

from config_detector import CONFIG as config, ENVIRONMENT
from utils.warning_suppressor import suppress_cuda_warnings
from utils.triton_warning_suppressor import suppress_triton_warnings, quiet_transcription

# Suppress CUDA/Triton warnings for cleaner output
suppress_cuda_warnings()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OfflineTranscriber:
    """Offline video transcription using Whisper."""
    
    def __init__(self):
        """Initialize transcriber with environment-specific optimization."""
        if ENVIRONMENT == "rtx_3060_local":
            # RTX 3060 CUDA optimization
            self.device = torch.device(config.CUDA_DEVICE)
            torch.cuda.set_per_process_memory_fraction(config.GPU_MEMORY_FRACTION)
            torch.cuda.empty_cache()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"Transcriber initialized on RTX 3060: {gpu_name} ({gpu_memory_gb:.1f}GB)")
        elif torch.cuda.is_available():
            # Generic CUDA
            self.device = torch.device("cuda")
            logger.info(f"Transcriber initialized on CUDA: {torch.cuda.get_device_name(0)}")
        else:
            # CPU fallback
            self.device = torch.device("cpu")
            logger.info("Transcriber initialized on CPU")
            
        self.model = None
        self.model_size = config.WHISPER_MODEL_SIZE
    
    def load_model(self, model_size: str = None, fast_mode: bool = True) -> bool:
        """
        Load Whisper model with GPU acceleration and memory management.
        
        Args:
            model_size: Model size ('tiny', 'base', 'small', 'medium', 'large')
            
        Returns:
            True if model loaded successfully
        """
        try:
            # CRITICAL FIX: Unload existing model first to prevent memory leaks
            if self.model is not None:
                logger.info("Unloading existing Whisper model to free memory")
                del self.model
                self.model = None
                torch.cuda.empty_cache()
                
            if model_size:
                self.model_size = model_size
            
            # SPEED OPTIMIZATION: Use fast mode by default
            if fast_mode and not model_size:
                self.model_size = "base"  # Much faster than large, 95%+ accuracy
                logger.info("Using fast mode: Whisper 'base' model for 4x speed improvement")
            
            # MEMORY MANAGEMENT: Check available memory before loading
            from utils.gpu_manager import GPUManager
            gpu_manager = GPUManager()
            
            if torch.cuda.is_available():
                memory_status = gpu_manager.check_memory_pressure()
                logger.info(f"Memory status before model load: {memory_status['usage_percent']:.1f}% used")
                
                # Perform cleanup if memory pressure is high
                if memory_status['pressure_level'] in ['high', 'critical']:
                    logger.warning(f"High memory pressure detected ({memory_status['pressure_level']}), performing cleanup")
                    gpu_manager.optimize_memory_usage(aggressive=True)
            
            logger.info(f"Loading Whisper model: {self.model_size}")
            
            # Load model with GPU support
            self.model = whisper.load_model(
                self.model_size,
                device=self.device,
                download_root=str(config.MODELS_DIR / "whisper")
            )
            
            logger.info(f"Model loaded successfully on {self.device}")
            return True
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA out of memory while loading model: {e}")
            # Emergency cleanup and retry with CPU
            if torch.cuda.is_available():
                gpu_manager = GPUManager()
                gpu_manager.emergency_memory_cleanup()
                # Switch to CPU fallback
                self.device = torch.device("cpu")
                logger.warning("Switching to CPU due to GPU memory constraints")
                return self.load_model(model_size)
            return False
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            return False
    
    def cleanup_model(self):
        """Clean up GPU memory by unloading the model."""
        if self.model is not None:
            logger.info("Cleaning up Whisper model to free GPU memory")
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
    
    @quiet_transcription
    def transcribe_video(self, video_path: str, language: str = None) -> List[Dict]:
        """
        Transcribe video file to text with timestamps.
        
        Args:
            video_path: Path to video file
            language: Language code for transcription
            
        Returns:
            List of transcription segments with timestamps
        """
        if not self.model:
            if not self.load_model():
                raise RuntimeError("Failed to load Whisper model")
        
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        logger.info(f"Starting transcription for: {video_path.name}")
        
        try:
            # Extract audio from video if needed
            audio_path = self._extract_audio(video_path)
            
            # RTX 3060 optimized transcription with robust error handling
            try:
                result = self.model.transcribe(
                    str(audio_path),
                    language=language or config.WHISPER_LANGUAGE,
                    verbose=False,  # Reduce verbose output that might cause encoding issues
                    word_timestamps=True,
                    fp16=ENVIRONMENT == "rtx_3060_local"  # Use FP16 on RTX 3060
                )
            except (UnicodeEncodeError, RuntimeError, Exception) as e:
                logger.warning(f"Transcription failed ({type(e).__name__}: {e}), retrying with fallback settings")
                # Fallback with simplified parameters for problematic files
                result = self.model.transcribe(
                    str(audio_path),
                    language=language or config.WHISPER_LANGUAGE,
                    verbose=False,
                    word_timestamps=False,
                    fp16=False,  # Disable FP16 for compatibility
                    condition_on_previous_text=False  # Prevent tensor reshape errors
                )
            
            # Process transcription results with encoding safety
            transcription_data = self._process_whisper_result(result)
            
            # Clean up temporary audio file
            if audio_path != video_path:
                audio_path.unlink(missing_ok=True)
            
            logger.info(f"Transcription completed: {len(transcription_data)} segments")
            return transcription_data
            
        except UnicodeEncodeError as ue:
            error_msg = f"Unicode encoding error for {video_path.name}: Problematic characters in audio/subtitle content"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from ue
        except Exception as e:
            logger.error(f"Transcription failed for {video_path}: {e}")
            raise
    
    def _extract_audio(self, video_path: Path) -> Path:
        """Extract audio from video file using ffmpeg."""
        try:
            # Check if input is already audio
            if video_path.suffix.lower() in ['.wav', '.mp3', '.m4a', '.flac']:
                return video_path
            
            # Create temporary audio file
            temp_dir = config.TEMP_DIR
            temp_dir.mkdir(exist_ok=True)
            
            audio_path = temp_dir / f"{video_path.stem}_audio.wav"
            
            # Get media stream information for track selection
            from core.media_selector import MediaTrackSelector
            selector = MediaTrackSelector()
            stream_info = selector.analyze_media_streams(video_path)
            
            # Use recommended audio track
            recommended_audio = stream_info.get('recommended_audio', 0)
            
            # Try multiple FFmpeg approaches for robust audio extraction
            success = False
            last_error = ""
            
            # Method 1: Specific audio track extraction
            # Fix: recommended_audio is the stream index (e.g. 1), not the audio-only index
            # Use -map 0:{stream_index} instead of 0:a:{audio_index}
            cmd_specific = [
                'ffmpeg',
                '-i', str(video_path),
                '-map', f'0:{recommended_audio}',  # Fixed: use stream index directly
                '-acodec', 'pcm_s16le',
                '-ac', '1',
                '-ar', '16000',
                '-af', 'volume=1.0',
                '-t', '3600',
                '-y',
                str(audio_path)
            ]
            
            logger.info(f"Extracting audio from track {recommended_audio} for transcription")
            
            try:
                result = subprocess.run(
                    cmd_specific,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    timeout=300
                )
                
                if result.returncode == 0 and audio_path.exists() and audio_path.stat().st_size > 0:
                    success = True
                else:
                    last_error = result.stderr
            except Exception as e:
                last_error = str(e)
            
            # Method 2: Simple audio extraction (no specific track)
            if not success:
                logger.warning("Specific track extraction failed, trying simple extraction")
                cmd_simple = [
                    'ffmpeg',
                    '-i', str(video_path),
                    '-vn',  # No video
                    '-acodec', 'pcm_s16le',
                    '-ac', '1',
                    '-ar', '16000',
                    '-t', '3600',
                    '-y',
                    str(audio_path)
                ]
                
                try:
                    result = subprocess.run(
                        cmd_simple,
                        capture_output=True,
                        text=True,
                        encoding='utf-8',
                        errors='replace',
                        timeout=300
                    )
                    
                    if result.returncode == 0 and audio_path.exists() and audio_path.stat().st_size > 0:
                        success = True
                    else:
                        last_error = result.stderr
                except Exception as e:
                    last_error = str(e)
            
            # Method 3: Most basic extraction (no processing)
            if not success:
                logger.warning("Simple extraction failed, trying basic extraction")
                cmd_basic = [
                    'ffmpeg',
                    '-i', str(video_path),
                    '-vn',
                    '-acodec', 'copy',
                    '-t', '3600',
                    '-y',
                    str(audio_path).replace('.wav', '.aac')  # Try different format
                ]
                
                basic_audio_path = Path(str(audio_path).replace('.wav', '.aac'))
                
                try:
                    result = subprocess.run(
                        cmd_basic,
                        capture_output=True,
                        text=True,
                        encoding='utf-8',
                        errors='replace',
                        timeout=300
                    )
                    
                    if result.returncode == 0 and basic_audio_path.exists() and basic_audio_path.stat().st_size > 0:
                        audio_path = basic_audio_path
                        success = True
                    else:
                        last_error = result.stderr
                except Exception as e:
                    last_error = str(e)
            
            if not success:
                raise RuntimeError(f"FFmpeg failed with all methods. Last error: {last_error}")
            
            # Verify audio file exists and has content
            if not audio_path.exists() or audio_path.stat().st_size == 0:
                raise RuntimeError("Audio extraction failed - empty or missing file")
            
            logger.info(f"Audio extracted to: {audio_path}")
            return audio_path
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("Audio extraction timed out")
        except FileNotFoundError:
            raise RuntimeError("FFmpeg not found. Please install FFmpeg.")
        except Exception as e:
            raise RuntimeError(f"Audio extraction failed: {e}")
    
    def _process_whisper_result(self, result: Dict) -> List[Dict]:
        """Process Whisper transcription result into standardized format."""
        transcription_data = []
        
        # Process segments
        for segment in result.get('segments', []):
            # Clean and encode text properly
            text = segment['text'].strip()
            # Replace problematic characters that can't be encoded
            text = text.encode('utf-8', errors='replace').decode('utf-8')
            
            segment_data = {
                'start_time': segment['start'],
                'end_time': segment['end'],
                'text': text,
                'confidence': segment.get('avg_logprob', 0.0),
                'segment_id': segment['id']
            }
            
            # Add word-level timestamps if available
            if 'words' in segment:
                segment_data['words'] = [
                    {
                        'word': word['word'].encode('utf-8', errors='replace').decode('utf-8'),
                        'start': word['start'],
                        'end': word['end'],
                        'probability': word.get('probability', 0.0)
                    }
                    for word in segment['words']
                ]
            
            transcription_data.append(segment_data)
        
        return transcription_data
    
    def transcribe_with_chunking(self, video_path: str, chunk_length: int = None) -> List[Dict]:
        """
        Transcribe long video by processing in chunks for memory efficiency.
        
        Args:
            video_path: Path to video file
            chunk_length: Length of chunks in seconds
            
        Returns:
            Combined transcription data
        """
        chunk_length = chunk_length or config.CHUNK_LENGTH_MS // 1000
        
        video_path = Path(video_path)
        
        # Get video duration
        duration = self._get_video_duration(video_path)
        
        if duration <= chunk_length:
            # Video is short enough to process in one go
            return self.transcribe_video(str(video_path))
        
        logger.info(f"Processing {duration:.1f}s video in {chunk_length}s chunks")
        
        all_transcriptions = []
        chunk_start = 0
        
        while chunk_start < duration:
            chunk_end = min(chunk_start + chunk_length, duration)
            
            logger.info(f"Processing chunk: {chunk_start:.1f}s - {chunk_end:.1f}s")
            
            # Extract chunk
            chunk_path = self._extract_video_chunk(video_path, chunk_start, chunk_end)
            
            try:
                # Transcribe chunk
                chunk_transcription = self.transcribe_video(str(chunk_path))
                
                # Adjust timestamps to global timeline
                for segment in chunk_transcription:
                    segment['start_time'] += chunk_start
                    segment['end_time'] += chunk_start
                    
                    if 'words' in segment:
                        for word in segment['words']:
                            word['start'] += chunk_start
                            word['end'] += chunk_start
                
                all_transcriptions.extend(chunk_transcription)
                
            finally:
                # Clean up chunk file
                chunk_path.unlink(missing_ok=True)
            
            chunk_start = chunk_end
        
        logger.info(f"Chunked transcription completed: {len(all_transcriptions)} total segments")
        return all_transcriptions
    
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
            
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=30)
            
            if result.returncode == 0:
                return float(result.stdout.strip())
            
        except (subprocess.TimeoutExpired, ValueError):
            pass
        
        # Fallback to OpenCV
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        
        return frame_count / fps if fps > 0 else 0
    
    def _extract_video_chunk(self, video_path: Path, start_time: float, end_time: float) -> Path:
        """Extract a chunk of video for processing."""
        temp_dir = config.TEMP_DIR
        chunk_path = temp_dir / f"{video_path.stem}_chunk_{start_time:.0f}_{end_time:.0f}.mp4"
        
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-ss', str(start_time),
            '-t', str(end_time - start_time),
            '-c', 'copy',
            '-y',
            str(chunk_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        
        if result.returncode != 0:
            raise RuntimeError(f"Video chunk extraction failed: {result.stderr}")
        
        return chunk_path
    
    def save_transcription(self, transcription_data: List[Dict], output_path: str) -> bool:
        """
        Save transcription data to file.
        
        Args:
            transcription_data: Transcription segments
            output_path: Output file path
            
        Returns:
            True if saved successfully
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as JSON for full data preservation
            if output_path.suffix.lower() == '.json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(transcription_data, f, indent=2, ensure_ascii=False)
            
            # Save as SRT for compatibility
            elif output_path.suffix.lower() == '.srt':
                self._save_as_srt(transcription_data, output_path)
            
            # Save as plain text
            else:
                with open(output_path, 'w', encoding='utf-8') as f:
                    for segment in transcription_data:
                        f.write(f"{segment['text']}\n")
            
            logger.info(f"Transcription saved to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save transcription: {e}")
            return False
    
    def _save_as_srt(self, transcription_data: List[Dict], output_path: Path):
        """Save transcription as SRT subtitle file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(transcription_data, 1):
                start_time = self._format_srt_time(segment['start_time'])
                end_time = self._format_srt_time(segment['end_time'])
                
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{segment['text']}\n\n")
    
    def _format_srt_time(self, seconds: float) -> str:
        """Format seconds as SRT timestamp."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        if not self.model:
            return {"status": "No model loaded"}
        
        return {
            "model_size": self.model_size,
            "device": str(self.device),
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "cuda_available": torch.cuda.is_available(),
            "gpu_memory_allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        }
