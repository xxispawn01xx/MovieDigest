# Overview

The Video Summarization Engine is an offline, GPU-accelerated application for processing movies and generating intelligent summaries with narrative analysis. The system discovers video files, extracts scenes using computer vision, transcribes audio using Whisper, analyzes narrative structure with local LLMs, and creates summaries with VLC bookmarks for easy navigation. Built with Streamlit for the web interface, it emphasizes local processing without external API dependencies.

# User Preferences

Preferred communication style: Simple, everyday language.

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