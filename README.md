# Video Summarization Engine

An advanced offline GPU-accelerated application for processing movies and generating intelligent summaries with narrative analysis. The system discovers video files, extracts scenes using computer vision, transcribes audio using Whisper, analyzes narrative structure with local LLMs, and creates summaries with VLC bookmarks for easy navigation.

## Features

### Core Capabilities
- **Offline Processing**: Complete offline operation using local Whisper and LLM models
- **GPU Acceleration**: CUDA-optimized processing for RTX 3060 and similar GPUs
- **Intelligent Scene Detection**: Multi-algorithm scene boundary detection
- **Audio Transcription**: OpenAI Whisper-powered speech-to-text
- **Narrative Analysis**: Local LLM integration for story structure understanding
- **Smart Summarization**: 5 different algorithms for optimal video compression

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
- CUDA-compatible GPU (recommended: RTX 3060 12GB or better)
- FFmpeg for video processing
- 16GB+ RAM recommended
- 50GB+ free disk space for models and processing

### Quick Start
1. **Clone and Install Dependencies**
   ```bash
   git clone <repository-url>
   cd video-summarization-engine
   ```
   Dependencies are automatically managed by Replit's package system.

2. **Configure GPU Settings**
   Edit `config.py` to match your GPU:
   ```python
   # For RTX 3060 12GB
   GPU_MEMORY_LIMIT_GB = 10
   CUDA_DEVICE = 0
   ```

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
- **RTX 3060 12GB**: Set GPU_MEMORY_LIMIT_GB = 10
- **RTX 4070 16GB**: Set GPU_MEMORY_LIMIT_GB = 14
- **RTX 4090 24GB**: Set GPU_MEMORY_LIMIT_GB = 20

### Processing Recommendations
- **Batch Size**: Start with 2 videos, increase if you have more VRAM
- **Whisper Models**: 
  - `base` (39 params) - Good balance of speed and accuracy
  - `small` (74M params) - Faster processing, lower accuracy
  - `medium` (769M params) - Higher accuracy, slower processing

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