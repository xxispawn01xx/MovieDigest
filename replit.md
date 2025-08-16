# Overview

The Video Summarization Engine is an offline, GPU-accelerated application designed to process movies and generate intelligent summaries with narrative analysis. It discovers video files, extracts scenes using computer vision, transcribes audio, analyzes narrative structure with local LLMs, and creates summaries with VLC bookmarks for easy navigation. Built with Streamlit, it emphasizes local processing without external API dependencies. The project aims to provide an efficient, high-quality video summarization solution for personal and professional use, with a focus on speed and local processing capabilities.

# User Preferences

Preferred communication style: Simple, everyday language.
Preferred LLM models: Mistral (uncensored) over traditional models like DialoGPT.

# System Architecture

## Frontend Architecture
- **Streamlit Web Interface**: Multi-page application with sidebar navigation, including Model Management, Export Controls, Email Marketing system, and real-time progress tracking.
- **State Management**: Session-based state management for user preferences, processing status, selected videos, and model configurations.
- **Visualization**: Plotly-based charts and graphs for displaying processing metrics, validation scores, and system status.
- **Model Management UI**: Interactive interface for downloading Whisper models, checking system requirements, and managing LLM models.

## Core Processing Pipeline
- **Video Discovery**: Scans directories for supported video formats (MP4, MKV, AVI, MOV, WMV) and extracts metadata.
- **Scene Detection**: Multi-algorithm approach using PySceneDetect with content-based, adaptive, and threshold detection methods.
- **Audio Transcription**: Offline speech-to-text using OpenAI's Whisper models with GPU acceleration and a 3-tier FFmpeg fallback system.
- **Narrative Analysis**: Local LLM integration for understanding film structure and identifying key narrative moments.
- **Video Summarization**: Combines all analysis to create compressed summaries targeting 15-20% of original length based on video duration.
- **VLC Bookmark Generation**: Creates XSPF playlist files with bookmarks for key scenes.
- **Export Management**: Multiple output formats including JSON summaries, VLC bookmarks, and markdown reports.
- **Model Management**: Automated download and management of Whisper models with intelligent recommendations.
- **Queue Management**: Comprehensive queue management with individual video removal, real-time progress, and bulk operations.
- **Speed Optimization**: Configurable optimization levels (standard, fast, ultra_fast) providing significant processing speed improvements while maintaining high transcription accuracy.

## Data Storage
- **SQLite Database**: Local database tracking video metadata, processing status, scene data, transcription results, and validation metrics.
- **File-based Storage**: Models stored locally, temporary files, and outputs.

## Advanced GPU Memory Management
- **Intelligent Memory Pressure Detection**: Real-time monitoring with automatic cleanup at various VRAM usage thresholds.
- **Multi-Tier Cleanup System**: Standard, aggressive, and emergency memory optimization with garbage collection and cache clearing.
- **CUDA Out-of-Memory Recovery**: Automatic detection and recovery from OOM errors with reduced processing settings.
- **Adaptive Batch Sizing**: Dynamic batch size adjustment based on video characteristics, available memory, and historical performance.
- **Sustained Processing Optimization**: Preventive cleanup and continuous memory monitoring for consistent performance.

## System Design Choices
- **Offline Processing**: Emphasis on local processing to ensure privacy and eliminate reliance on external APIs.
- **GPU Acceleration**: Utilizes NVIDIA GPUs (specifically optimized for RTX 3060) for accelerated processing.
- **Robust Error Handling**: Implemented comprehensive error handling for audio extraction, tensor reshape failures, and OOM issues, ensuring continued processing of remaining videos.
- **VLC Playback Compatibility**: Ensures summaries and generated files are fully compatible with VLC for seamless playback and seeking.
- **Intelligent Track Analysis**: Detects and selects optimal audio tracks and subtitle streams.
- **Custom Output Directory**: Allows users to select custom output locations.
- **Credits Detection**: Advanced computer vision-based credits detection system.

# External Dependencies

## AI/ML Models
- **Whisper**: OpenAI's speech recognition models for offline transcription.
- **Local LLM**: Transformer-based language models for narrative analysis (user-provided).
- **PyTorch**: Deep learning framework with CUDA support for GPU acceleration.

## Video Processing Libraries
- **OpenCV**: Computer vision library for video analysis and frame processing.
- **PySceneDetect**: Scene boundary detection.
- **FFmpeg**: Video processing and metadata extraction via subprocess calls.

## Web Framework
- **Streamlit**: Web application framework for the user interface.
- **Plotly**: Interactive visualization library for charts and progress displays.

## System Libraries
- **SQLite3**: Embedded database for local data persistence.
- **psutil**: System resource monitoring and process management.
- **scikit-learn**: Machine learning utilities for validation metrics.
- **scipy**: Scientific computing library for statistical analysis.

## File Format Support
- **Video Formats**: MP4, MKV, AVI, MOV, WMV.
- **Subtitle Formats**: SRT, VTT, ASS, SSA.
- **Output Formats**: XSPF playlists, JSON metadata, video clips.