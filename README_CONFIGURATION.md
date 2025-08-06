# Configuration Guide

This document explains how to configure the Video Summarization Engine for different setups and requirements, including the new RTX 3060 automatic optimization system. - Video Summarization Engine

## Environment Variables

You can control the behavior of the application using environment variables. Set these before starting the app to customize its operation.

### Development/Testing Configuration

#### Disable Automatic Downloads
```bash
export DISABLE_AUTO_DOWNLOADS=true
streamlit run app.py --server.port 5000
```
This prevents the automatic downloading of Whisper models during startup.

#### Disable Database Initialization  
```bash
export DISABLE_DATABASE_INIT=true
streamlit run app.py --server.port 5000
```
This skips automatic database table creation and updates.

#### Enable Offline Mode
```bash
export OFFLINE_MODE=true
streamlit run app.py --server.port 5000
```
This disables all network operations including model downloads and external API calls.

#### Combined Development Mode
```bash
export DISABLE_AUTO_DOWNLOADS=true
export DISABLE_DATABASE_INIT=true
export OFFLINE_MODE=true
streamlit run app.py --server.port 5000
```

### GPU Configuration

```bash
export CUDA_VISIBLE_DEVICES=0
export MAX_GPU_MEMORY_GB=10
```

### Model Configuration

```bash
export WHISPER_MODEL=base
```

### Processing Configuration

```bash
export DEFAULT_BATCH_SIZE=2
export MAX_BATCH_SIZE=5
```

## Configuration File

You can also create a `.env` file in the project root with your settings:

```bash
cp .env.example .env
# Edit .env with your preferences
```

## Visual Indicators

When development flags are active, you'll see indicators in the sidebar:
- ⚙️ Development Mode Active
- 🚫 Auto-downloads disabled
- 🚫 Database init disabled
- 📴 Offline mode enabled

## Use Cases

### Testing Without Network Access
```bash
export OFFLINE_MODE=true
```

### Development Without Database Changes
```bash
export DISABLE_DATABASE_INIT=true
```

### Prevent Model Downloads
```bash
export DISABLE_AUTO_DOWNLOADS=true
```

### Complete Isolation
```bash
export DISABLE_AUTO_DOWNLOADS=true
export DISABLE_DATABASE_INIT=true
export OFFLINE_MODE=true
```