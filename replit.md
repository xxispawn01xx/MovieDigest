# Overview

The Video Summarization Engine is an offline, GPU-accelerated application for processing movies and generating intelligent summaries with narrative analysis. The system discovers video files, extracts scenes using computer vision, transcribes audio using Whisper, analyzes narrative structure with local LLMs, and creates summaries with VLC bookmarks for easy navigation. Built with Streamlit for the web interface, it emphasizes local processing without external API dependencies.

# User Preferences

Preferred communication style: Simple, everyday language.

# Recent Changes

## Latest Updates (Aug 5, 2025)
- **Recent Folders Feature**: Fully implemented recent folder tracking with database integration, UI buttons for quick access, and automatic directory tracking
- **Encoding Fix**: Resolved Unicode/character encoding issues in transcription that caused processing failures on Windows systems
- **GitIgnore Enhancement**: Created comprehensive offline-only gitignore structure to keep video databases local while syncing code to remote
- **Database Integration**: Recent folders feature automatically tracks scanned directories with video count, scan duration, and metadata
- **UI Polish**: Recent folders display with quick access buttons, expandable sections, and remove functionality
- **Enhanced Export System**: Added comprehensive Export Manager with JSON, VLC bookmarks, and markdown report generation
- **Model Management**: Built intelligent Model Downloader with Whisper model recommendations and automated downloads
- **UI Improvements**: Created Model Management page with system requirements checking and download interface
- **VLC Integration**: Added specialized VLC bookmark generator for creating chapter-based and key moment playlists
- **Plex Integration**: Built complete Plex Media Server integration for genre filtering, rating-based sorting, and rich metadata discovery
- **Advanced Scene Analysis**: Created sophisticated scene characterization with visual and emotional analysis
- **Audio Analysis**: Built comprehensive audio feature extraction including speech detection and music analysis
- **Smart Summarization**: Implemented multi-algorithm summarization engine with hybrid, narrative, and importance-based methods
- **Documentation**: Created comprehensive README.md with architecture overview, installation guide, and usage instructions
- **Processing Queue Fix**: Resolved issue where "Add to Queue" button didn't start processing by connecting database status with internal queue  
- **Batch Page Enhancement**: Added queued videos display to show database content on processing page
- **Configuration System**: Added environment variables to control auto-downloads, database init, and offline mode
- **Development Mode**: Added visual indicators when development flags are active
- **Fallback Mode**: Ensured app works without transformers library using intelligent fallback analysis
- **Credits Detection Enhancement**: Built advanced computer vision-based credits detection system with visual pattern analysis, text density detection, audio analysis, and fade transition detection
- **Subtitle Preservation**: Fixed FFmpeg commands to preserve original video subtitles in summary outputs
- **VLC Auto-Detection**: Implemented proper VLC bookmark naming convention for automatic detection when opening videos
- **Custom Output Directory**: Added GUI interface for selecting custom output directories in Settings page

# System Architecture

## Frontend Architecture
- **Streamlit Web Interface**: Multi-page application with sidebar navigation including Model Management, Export Controls, and real-time progress tracking
- **State Management**: Session-based state management for user preferences, processing status, selected videos, and model configurations
- **Visualization**: Plotly-based charts and graphs for displaying processing metrics, validation scores, and system status
- **Model Management UI**: Interactive interface for downloading Whisper models, checking system requirements, and managing LLM models

## Core Processing Pipeline
- **Video Discovery**: Scans directories for supported video formats (MP4, MKV, AVI, MOV, WMV) and extracts metadata using OpenCV and ffprobe
- **Scene Detection**: Multi-algorithm approach using PySceneDetect with content-based, adaptive, and threshold detection methods
- **Audio Transcription**: Offline speech-to-text using OpenAI's Whisper models with GPU acceleration
- **Narrative Analysis**: Local LLM integration for understanding film structure and identifying key narrative moments (with fallback mode)
- **Video Summarization**: Combines all analysis to create compressed summaries targeting 15% of original length
- **VLC Bookmark Generation**: Creates XSPF playlist files with bookmarks for key scenes
- **Export Management**: Multiple output formats including JSON summaries, VLC bookmarks, and markdown reports
- **Model Management**: Automated download and management of Whisper models with intelligent recommendations

## Data Storage
- **SQLite Database**: Local database tracking video metadata, processing status, scene data, transcription results, and validation metrics
- **File-based Storage**: Models stored locally in `/models` directory, temporary files in `/temp`, and outputs in `/output`

## GPU Management
- **CUDA Optimization**: Intelligent GPU memory management with configurable limits (default 10GB on RTX 3060)
- **Resource Monitoring**: Real-time GPU utilization, temperature, and memory usage tracking
- **Batch Processing**: Queue-based processing with automatic resource allocation and error recovery

## Validation System
- **Real-time Metrics**: F1-score calculation using TVSum/SumMe benchmark methodologies
- **Quality Assessment**: Precision, recall, and other validation metrics calculated during processing
- **Synthetic Ground Truth**: Generates benchmark comparisons when human annotations unavailable

# External Dependencies

## AI/ML Models
- **Whisper**: OpenAI's speech recognition models for offline transcription
- **Local LLM**: Transformer-based language models for narrative analysis (user-provided)
- **PyTorch**: Deep learning framework with CUDA support for GPU acceleration

## Video Processing Libraries
- **OpenCV**: Computer vision library for video analysis and frame processing
- **PySceneDetect**: Scene boundary detection with multiple algorithm support
- **FFmpeg**: Video processing and metadata extraction via subprocess calls

## Web Framework
- **Streamlit**: Web application framework for the user interface
- **Plotly**: Interactive visualization library for charts and progress displays

## System Libraries
- **SQLite3**: Embedded database for local data persistence
- **psutil**: System resource monitoring and process management
- **scikit-learn**: Machine learning utilities for validation metrics
- **scipy**: Scientific computing library for statistical analysis

## File Format Support
- **Video Formats**: MP4, MKV, AVI, MOV, WMV
- **Subtitle Formats**: SRT, VTT, ASS, SSA
- **Output Formats**: XSPF playlists, JSON metadata, video clips