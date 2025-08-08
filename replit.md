# Overview

The Video Summarization Engine is an offline, GPU-accelerated application for processing movies and generating intelligent summaries with narrative analysis. The system discovers video files, extracts scenes using computer vision, transcribes audio using Whisper, analyzes narrative structure with local LLMs, and creates summaries with VLC bookmarks for easy navigation. Built with Streamlit for the web interface, it emphasizes local processing without external API dependencies.

# User Preferences

Preferred communication style: Simple, everyday language.
Preferred LLM models: Mistral (uncensored) over traditional models like DialoGPT.

# Recent Changes

## Latest Updates (Aug 8, 2025)
- **Critical Audio Extraction Fix**: Fixed "Failed to set value '0:a:1' for option 'map': Invalid argument" error by correcting FFmpeg stream mapping from `0:a:{index}` to `0:{index}` in transcription.py, summarizer.py, and media_selector.py
- **VLC Playback Compatibility**: Completely rewrote multi-scene video concatenation to use individual clip creation + concat demuxer instead of filter_complex, ensuring perfect VLC seeking and playback without errors
- **Enhanced Video Output**: Added proper audio track mapping for single and multi-scene summaries with `-movflags '+faststart'` for optimal VLC compatibility and seeking performance
- **Comprehensive GPU Memory Management**: Implemented proactive CUDA out-of-memory detection and prevention system with intelligent memory pressure monitoring, automatic model cleanup between videos, and multi-tier emergency recovery for sustained RTX 3060 processing
- **Duplicate Processing Prevention**: Fixed critical batch processing bug where videos were processed multiple times by enhancing status tracking logic and ensuring proper completion marking to prevent queue duplication
- **Intelligent Model Lifecycle Management**: Added automatic Whisper model unloading and GPU memory cleanup after each transcription to prevent memory leaks and fragmentation issues that caused processing failures

## Previous Updates (Aug 6, 2025)
- **Hugging Face Token Integration**: Added secure HF_TOKEN configuration in Model Management page with one-click authenticated model downloads, token management interface with save/clear functionality, and automatic token integration with download commands for seamless access to gated models
- **CUDA Memory Management Enhancement**: Implemented comprehensive CUDA out-of-memory prevention system with intelligent memory pressure detection, multi-tier cleanup (standard/aggressive/emergency), automatic OOM recovery with reduced settings, and sustained processing optimization for RTX 3060 systems
- **Robust Audio Extraction**: Enhanced audio extraction with 3-tier FFmpeg fallback system to handle complex MKV files with encoding issues, ensuring reliable transcription for all video formats
- **Windows Encoding Fix**: Resolved Unicode encoding issue preventing Mistral model downloads on Windows systems by implementing comprehensive encoding fix utility with UTF-8 forcing and fallback handling
- **Directory Scanning Fix**: Fixed "object of type 'generator' has no len()" error in video discovery by properly converting generator to list before length calculation
- **Email Marketing System**: Built comprehensive email outreach system based on RADflow project patterns, featuring entertainment industry templates, prospect database with 40+ verified contacts from major studios/streaming platforms, campaign management with SendGrid integration, and professional email templates targeting production companies, post-production facilities, and content acquisition teams
- **One-Click LLM Model Downloads**: Created enhanced Model Management interface with one-click installation buttons for recommended models like Mistral-7B-Instruct (uncensored), including progress tracking, system requirements checking, and the exact huggingface-cli commands integrated as clickable buttons
- **Audio Track & Subtitle Selection System**: Implemented comprehensive MediaTrackSelector that automatically detects and selects optimal audio tracks (default/English/most channels priority) and subtitles (SRT, VTT, ASS, SSA) from video files during processing
- **Enhanced Video Processing**: Updated transcription and summarization to use selected audio tracks, fixing issues where users couldn't access multiple audio streams or subtitle tracks from original videos
- **Streamlit Navigation Fix**: Fixed problematic st.switch_page calls causing navigation errors by properly structuring pages directory and using correct page routing
- **Intelligent Track Analysis**: Built FFprobe-based stream analyzer that identifies video codec, audio tracks with language/channel info, and subtitle streams with format detection
- **External Subtitle Support**: Added automatic detection of external subtitle files with language-specific naming patterns (e.g., .en.srt, .spanish.srt)
- **Batch Media Settings**: Created interface for configuring audio/subtitle selection strategies across multiple videos in batch processing
- **Adaptive Compression System**: Implemented smart compression ratios - 15% for videos ≤90 minutes, 20% for longer videos to maintain better narrative flow
- **Critical Summarization Fixes**: Fixed core issues preventing proper video summarization including scene selection validation, FFmpeg command optimization, and strict compression enforcement
- **Video Seeking Fix**: Implemented proper FFmpeg concatenation with faststart flag and timestamp reset for seeking compatibility
- **Scene Selection Enhancement**: Added validation to prevent invalid scene data and ensure proper duration constraints
- **Enhanced Pause/Resume System**: Added comprehensive pause/resume controls with separate pause (temporary) vs stop (complete) functionality for better batch processing control
- **Queue Management Fix**: Fixed GUI update issue where clearing the queue updated console but not the interface - now properly syncs memory and database
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
- **Auto-Loading Video Discovery**: Videos now auto-populate from database on page load instead of requiring manual "Scan Directory" clicks for previously discovered videos
- **Individual Video Selection**: Complete rebuild of video selection interface with individual checkboxes, "Select All/Clear All" controls, selection counter, and smart "Add Selected (X)" button functionality

# System Architecture

## Frontend Architecture
- **Streamlit Web Interface**: Multi-page application with sidebar navigation including Model Management, Export Controls, Email Marketing system, and real-time progress tracking
- **State Management**: Session-based state management for user preferences, processing status, selected videos, and model configurations
- **Visualization**: Plotly-based charts and graphs for displaying processing metrics, validation scores, and system status
- **Model Management UI**: Interactive interface for downloading Whisper models, checking system requirements, and managing LLM models

## Core Processing Pipeline
- **Video Discovery**: Scans directories for supported video formats (MP4, MKV, AVI, MOV, WMV) and extracts metadata using OpenCV and ffprobe
- **Scene Detection**: Multi-algorithm approach using PySceneDetect with content-based, adaptive, and threshold detection methods
- **Audio Transcription**: Offline speech-to-text using OpenAI's Whisper models with GPU acceleration and 3-tier FFmpeg fallback system
- **Narrative Analysis**: Local LLM integration for understanding film structure and identifying key narrative moments (with fallback mode)
- **Video Summarization**: Combines all analysis to create compressed summaries targeting 15% of original length
- **VLC Bookmark Generation**: Creates XSPF playlist files with bookmarks for key scenes
- **Export Management**: Multiple output formats including JSON summaries, VLC bookmarks, and markdown reports
- **Model Management**: Automated download and management of Whisper models with intelligent recommendations

## Data Storage
- **SQLite Database**: Local database tracking video metadata, processing status, scene data, transcription results, and validation metrics
- **File-based Storage**: Models stored locally in `/models` directory, temporary files in `/temp`, and outputs in `/output`

## Advanced GPU Memory Management
- **Intelligent Memory Pressure Detection**: Real-time monitoring with automatic cleanup at 70%, 80%, and 90% VRAM usage thresholds
- **Multi-Tier Cleanup System**: Standard, aggressive, and emergency memory optimization with garbage collection and cache clearing
- **CUDA Out-of-Memory Recovery**: Automatic detection and recovery from OOM errors with reduced processing settings and emergency mode fallback
- **Adaptive Batch Sizing**: Dynamic batch size adjustment based on video characteristics, available memory, and historical performance
- **Memory-Conservative Processing**: Emergency processing mode with 75% reduced batch sizes and optimized chunk lengths
- **Sustained Processing Optimization**: Preventive cleanup every 2 videos, continuous memory monitoring, and intelligent resource allocation

## RTX 3060 Optimization Settings
- **Memory Allocation**: 11GB VRAM usage (85% of 12.9GB total) with 1.9GB system reserve
- **Recommended Batch Sizes**: Video processing 1-2 videos, Whisper transcription 4-6 chunks, LLM processing 2-4 batches
- **Performance Features**: Tensor Core acceleration, TF32 optimization, mixed precision training, cuDNN benchmarking
- **Emergency Mode Settings**: Whisper batch 2-4, LLM batch 1-2, reduced chunk lengths for memory recovery

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