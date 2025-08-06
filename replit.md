# Overview

The Video Summarization Engine is an offline, GPU-accelerated application for processing movies and generating intelligent summaries with narrative analysis. The system discovers video files, extracts scenes using computer vision, transcribes audio using Whisper, analyzes narrative structure with local LLMs, and creates summaries with VLC bookmarks for easy navigation. Built with Streamlit for the web interface, it emphasizes local processing without external API dependencies.

# User Preferences

Preferred communication style: Simple, everyday language.

# Recent Changes

## Latest Updates (Aug 6, 2025)
- **Adaptive Compression System**: Implemented smart compression ratios - 15% for videos ≤90 minutes, 20% for longer videos to maintain better narrative flow
- **Critical Summarization Fixes**: Fixed core issues preventing proper video summarization including scene selection validation, FFmpeg command optimization, and strict compression enforcement
- **Video Seeking Fix**: Implemented proper FFmpeg concatenation with faststart flag and timestamp reset for seeking compatibility
- **Streamlit Navigation Fix**: Replaced problematic st.switch_page calls with st.rerun to prevent app crashes
- **Scene Selection Enhancement**: Added validation to prevent invalid scene data and ensure proper duration constraints
- **Error Handling Improvement**: Eliminated fallback video copying that was causing full-length outputs instead of summaries
- **Windows Unicode Fix**: Resolved Windows Command Prompt encoding issues with RTX 3060 configuration detection
- **RTX 3060 Optimization**: Created smart environment detection system that automatically configures for RTX 3060 when running locally vs CPU fallback on Replit cloud
- **Batch Processing Robustness**: Added advanced error handling for tensor reshape failures and audio extraction issues that caused video processing crashes
- **Triton Warning Suppression**: Implemented warning suppressor to eliminate cosmetic Triton kernel warnings on RTX 3060 systems
- **Audio Extraction Enhancement**: Added validation and normalization for audio extraction to prevent empty file failures
- **Whisper Fallback System**: Created robust fallback transcription with simplified parameters for problematic video files
- **Environment-Specific Config**: RTX 3060 uses large Whisper model, 85% VRAM, FP16 acceleration; Replit uses CPU-optimized settings
- **Memory Management**: Optimized GPU memory allocation with 11GB usage on RTX 3060, automatic cleanup, and batch size optimization
- **Processing Resilience**: System continues processing remaining videos when individual files fail rather than stopping entire batch
- **Recent Folders Feature**: Fully implemented recent folder tracking with database integration, UI buttons for quick access, and automatic directory tracking
- **Encoding Fix**: Resolved Unicode/character encoding issues in transcription that caused processing failures on Windows systems
- **VLC Auto-Detection**: Implemented proper VLC bookmark naming convention for automatic detection when opening videos
- **Custom Output Directory**: Added GUI interface for selecting custom output directories in Settings page
- **Credits Detection Enhancement**: Built advanced computer vision-based credits detection system with visual pattern analysis, text density detection, audio analysis, and fade transition detection
- **Enhanced Export System**: Added comprehensive Export Manager with JSON, VLC bookmarks, and markdown report generation
- **Model Management**: Built intelligent Model Downloader with Whisper model recommendations and automated downloads
- **Plex Integration**: Built complete Plex Media Server integration for genre filtering, rating-based sorting, and rich metadata discovery
- **Processing Queue Fix**: Resolved issue where "Add to Queue" button didn't start processing by connecting database status with internal queue

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