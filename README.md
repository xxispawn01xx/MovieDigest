# Video Summarization Engine

An advanced offline GPU-accelerated application for processing movies and generating intelligent summaries with narrative analysis. The system discovers video files, extracts scenes using computer vision, transcribes audio using Whisper, analyzes narrative structure with local LLMs, and creates summaries with VLC bookmarks for easy navigation.

## Features

### Core Capabilities
- **Offline Processing**: Complete offline operation using local Whisper and LLM models
- **RTX 3060 Optimization**: Smart environment detection with automatic GPU configuration and FP16 acceleration
- **Intelligent Scene Detection**: Multi-algorithm scene boundary detection with GPU acceleration
- **Audio Transcription**: OpenAI Whisper-powered speech-to-text with robust error handling
- **Narrative Analysis**: Local LLM integration for story structure understanding with CPU fallback
- **Smart Summarization**: 5 different algorithms for optimal video compression with batch processing

### Advanced Analysis
- **Scene Characterization**: Visual complexity, motion analysis, and emotional content detection
- **Audio Analysis**: Speech detection, music recognition, and sound pattern analysis
- **Multi-Algorithm Summarization**: Hybrid, importance-based, narrative structure, audio-visual sync, and emotional flow
- **Quality Validation**: F1-score calculation using TVSum/SumMe methodologies

### Integration & Export
- **Plex Media Server**: Genre filtering, rating-based sorting, and metadata discovery
- **VLC Bookmarks**: Chapter-based and key moment playlist generation
- **Export Formats**: JSON summaries, VLC bookmarks, markdown reports, and CSV exports
- **Model Management**: Automated Whisper model downloads with intelligent recommendations

## Architecture

### Frontend (Streamlit Web Interface)
```
app.py - Main application with multi-page navigation
├── Overview - Dashboard and system status
├── Video Discovery - Local file scanning and metadata extraction
├── Plex Integration - Media server connectivity and filtering
├── Advanced Analysis - Smart summarization with multiple algorithms
├── Processing Queue - Real-time processing status and controls
├── Batch Processing - Multi-video processing with resource management
├── Summary Results - Visualization and analysis results
├── Export Center - Multiple output format generation
├── Model Management - Whisper model downloads and system requirements
└── System Status - GPU monitoring and performance metrics
```

### Core Processing Pipeline
```
core/
├── video_discovery.py - Video file scanning and metadata extraction
├── scene_detector.py - Multi-algorithm scene boundary detection
├── transcription.py - Offline Whisper audio transcription
├── narrative_analyzer.py - Local LLM narrative structure analysis
├── advanced_scene_analysis.py - Visual characteristics and emotional analysis
├── audio_analysis.py - Speech detection and music recognition
├── smart_summarization.py - Multi-algorithm summarization engine
├── summarizer.py - Core video summarization logic
├── validation.py - Quality metrics and F1-score calculation
├── batch_processor.py - Multi-video processing with GPU management
├── export_manager.py - Multiple output format generation
├── model_downloader.py - Automated model management
├── plex_integration.py - Media server connectivity
├── vlc_bookmarks.py - XSPF playlist generation
└── database.py - SQLite data persistence
```

### Data Storage
```
├── video_processing.db - SQLite database for metadata and results
├── models/
│   ├── whisper/ - Whisper model files
│   └── llm/ - Local language models
├── output/
│   ├── bookmarks/ - VLC XSPF playlists
│   ├── summaries/ - JSON and markdown reports
│   └── exports/ - CSV and other export formats
└── temp/ - Temporary processing files
```

### Utilities
```
utils/
├── gpu_manager.py - CUDA memory management and monitoring
└── progress_tracker.py - Real-time processing progress updates
```

## Installation & Setup

### Prerequisites
- Python 3.11 or higher
- CUDA-compatible GPU (optimized for RTX 3060 12GB, auto-detects environment)
- FFmpeg for video processing
- 16GB+ RAM recommended
- 50GB+ free disk space for models and processing

### Environment Support
- **RTX 3060 Local**: Full GPU optimization with large Whisper model, 85% VRAM usage, FP16 acceleration
- **Generic CUDA**: Automatic detection with conservative memory settings
- **CPU Fallback**: Replit cloud compatibility with CPU-only processing

### Quick Start
1. **Clone and Install Dependencies**
   ```bash
   git clone <repository-url>
   cd video-summarization-engine
   ```
   Dependencies are automatically managed by Replit's package system.

2. **Environment Detection**
   The app automatically detects your environment:
   - **RTX 3060**: Uses `config_rtx3060.py` with optimized settings
   - **Other CUDA GPUs**: Uses standard configuration
   - **CPU-only**: Fallback mode for development/testing

3. **Download Whisper Models**
   Use the Model Management page in the web interface or manually:
   ```bash
   # The app will guide you through model downloads
   # Recommended: whisper-base or whisper-small for most users
   ```

4. **Start the Application**
   ```bash
   streamlit run app.py --server.port 5000
   ```

5. **Access the Web Interface**
   Open your browser to: `http://localhost:5000`
   
   **Note**: The server binds to `0.0.0.0:5000` but you access it via localhost for offline use.

## Usage Guide

### Basic Workflow
1. **Video Discovery**: Scan your local directories for video files
2. **Plex Integration** (Optional): Connect to your Plex server for enhanced metadata
3. **Processing Queue**: Add videos to the processing queue
4. **Batch Processing**: Start processing with GPU acceleration
5. **Advanced Analysis**: Generate smart summaries using multiple algorithms
6. **Export Results**: Download VLC bookmarks, JSON summaries, or reports

### Processing Pipeline
Each video goes through these stages:
1. **Metadata Extraction**: Duration, resolution, codec information
2. **Scene Detection**: Boundary detection using multiple algorithms
3. **Advanced Scene Analysis**: Visual characteristics and emotional content
4. **Audio Analysis**: Speech detection and music recognition
5. **Transcription**: Whisper-powered speech-to-text
6. **Narrative Analysis**: Story structure understanding (optional)
7. **Smart Summarization**: Multi-algorithm summary generation
8. **Validation**: Quality metrics calculation
9. **Export Generation**: Multiple output formats

### Summarization Algorithms

#### 1. Hybrid (Recommended)
Combines multiple analysis methods for optimal results:
- Narrative structure (30%)
- Scene importance (40%)
- Audio-visual sync (20%)
- Emotional flow (10%)

#### 2. Importance-Based
Selects scenes with highest importance scores based on:
- Visual complexity
- Motion intensity
- Audio characteristics
- Position in narrative

#### 3. Narrative Structure
Follows 5-act story structure:
- Opening (10%)
- Inciting Incident (10-25%)
- Rising Action (25-50%)
- Climax (50-75%)
- Resolution (75-100%)

#### 4. Audio-Visual Sync
Focuses on peak audio-visual moments:
- High motion with sound
- Dialogue scenes
- Music montages
- Action sequences

#### 5. Emotional Flow
Maintains emotional progression:
- Emotional peaks
- Character development
- Mood transitions
- Dramatic intensity

## Configuration

### Core Settings (`config.py`)
```python
# GPU Configuration
GPU_MEMORY_LIMIT_GB = 10        # Adjust for your GPU
CUDA_DEVICE = 0                 # GPU device index

# Processing Settings
DEFAULT_BATCH_SIZE = 2          # Videos processed simultaneously
MAX_CONCURRENT_JOBS = 4         # Maximum parallel processing

# Model Paths
WHISPER_MODEL_PATH = Path("models/whisper")
LLM_MODEL_PATH = Path("models/llm")

# Output Settings
OUTPUT_DIR = Path("output")
TEMP_DIR = Path("temp")
```

### Streamlit Configuration (`.streamlit/config.toml`)
```toml
[server]
headless = true
address = "0.0.0.0"
port = 5000

# Note: Server binds to 0.0.0.0 for compatibility but access via localhost:5000

[theme]
base = "dark"
```

## Performance Optimization

### GPU Memory Management
- **RTX 3060 12GB**: Automatic 85% VRAM usage (11GB) with 1.9GB system reserve
- **Generic CUDA**: Conservative 8GB limit with auto-detection
- **CPU Mode**: No GPU memory usage for development environments

### Batch Processing Resilience
- **Error Recovery**: System continues processing remaining videos when individual files fail
- **Tensor Error Handling**: Robust fallback for problematic audio files with reshape errors
- **Audio Validation**: Prevents empty audio file failures with normalization
- **Triton Warning Suppression**: Clean logs without cosmetic CUDA toolkit warnings

### Processing Recommendations
- **RTX 3060 Batch Size**: Optimized for 3-4 videos simultaneously
- **Whisper Models**: 
  - **RTX 3060**: Uses `large` model automatically for best accuracy
  - `base` (74M params) - Good balance for standard GPUs
  - `small` (39M params) - Faster processing, lower accuracy
  - `medium` (769M params) - Higher accuracy, moderate speed

### Error Handling Improvements
- **Transcription Fallback**: Automatic retry with simplified parameters for problematic files
- **Audio Extraction**: Enhanced validation with volume normalization and time limits
- **Memory Management**: Intelligent cleanup and garbage collection during batch processing

### Content-Specific Settings
- **Action Films**: Use Audio-Visual Sync algorithm
- **Dramas**: Use Emotional Flow algorithm  
- **Documentaries**: Use Importance-Based algorithm
- **TV Episodes**: Use Narrative Structure algorithm

## Export Formats

### VLC Bookmarks (XSPF)
- Chapter-based navigation
- Key moment timestamps
- Scene descriptions
- Compatible with VLC media player

### JSON Summaries
```json
{
  "video_info": {...},
  "scenes": [...],
  "narrative_analysis": {...},
  "summary": {...},
  "validation_metrics": {...}
}
```

### Markdown Reports
- Human-readable summaries
- Scene breakdowns
- Processing statistics
- Quality metrics

### CSV Exports
- Timestamp lists
- Scene data
- Processing results
- Compatible with Excel/Sheets

## Troubleshooting

### Common Issues

**RTX 3060 Configuration Issues**
```bash
# Check CUDA installation
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Environment will auto-detect RTX 3060 and apply optimizations
# Check logs for: "RTX 3060 configuration loaded"
```

**Batch Processing Errors**
- **Tensor Reshape Errors**: Now handled automatically with fallback transcription
- **Audio Extraction Failures**: Enhanced validation prevents empty file errors
- **Memory Issues**: Automatic cleanup and conservative batch sizing

**GPU Not Detected**
```bash
# Check CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

**Out of Memory Errors**
- Reduce GPU_MEMORY_LIMIT_GB
- Decrease batch size
- Use smaller Whisper model

**Slow Processing**
- Ensure GPU acceleration is enabled
- Check system resources (CPU, RAM)
- Verify video codec compatibility

**Model Download Issues**
- Check internet connection
- Verify disk space availability
- Use Model Management page for guided downloads

### Performance Monitoring
The System Status page provides real-time monitoring:
- GPU utilization and temperature
- Memory usage (VRAM and RAM)
- Processing queue status
- Error logs and diagnostics

## Development

### Project Structure
- `app.py` - Main Streamlit application
- `core/` - Core processing modules
- `utils/` - Utility functions and helpers
- `config.py` - Configuration settings
- `models/` - AI model storage
- `output/` - Generated outputs

### Adding New Features
1. Create module in `core/` directory
2. Add imports to `app.py`
3. Integrate with batch processor
4. Add UI components to relevant pages
5. Update documentation

## License

This project is designed for personal use with local video collections. Ensure you have appropriate rights to process video content.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review system requirements
3. Verify GPU and CUDA setup
4. Check processing logs in System Status

---

Built with Python, Streamlit, PyTorch, OpenAI Whisper, and OpenCV for offline video analysis and summarization.