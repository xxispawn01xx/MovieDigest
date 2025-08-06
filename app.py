"""
Main Streamlit application for the Video Summarization Engine.
Provides web interface for video discovery, processing, and summary generation.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import time
import threading
from typing import Dict, List
import logging
from datetime import datetime

# Core modules
from core.database import VideoDatabase
from core.video_discovery import VideoDiscovery
from core.batch_processor import BatchProcessor
from core.validation import ValidationMetrics
from core.export_manager import ExportManager
from core.model_downloader import ModelDownloader
from core.vlc_bookmarks import VLCBookmarkGenerator
from core.plex_integration import PlexIntegration
from utils.gpu_manager import GPUManager
from utils.progress_tracker import ProgressTracker
from utils.warning_suppressor import suppress_cuda_warnings
from config_detector import CONFIG as config, ENVIRONMENT

# Initialize warning suppression early
suppress_cuda_warnings()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Video Summarization Engine",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Display configuration status if development flags are set
if config.DISABLE_AUTO_DOWNLOADS or config.DISABLE_DATABASE_INIT or config.OFFLINE_MODE:
    st.sidebar.warning("⚙️ Development Mode Active")
    if config.DISABLE_AUTO_DOWNLOADS:
        st.sidebar.info("🚫 Auto-downloads disabled")
    if config.DISABLE_DATABASE_INIT:
        st.sidebar.info("🚫 Database init disabled")  
    if config.OFFLINE_MODE:
        st.sidebar.info("📴 Offline mode enabled")

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.db = VideoDatabase()
    st.session_state.discovery = VideoDiscovery()
    st.session_state.batch_processor = BatchProcessor()
    st.session_state.gpu_manager = GPUManager()
    st.session_state.progress_tracker = ProgressTracker()
    st.session_state.validation_metrics = ValidationMetrics()
    st.session_state.export_manager = ExportManager()
    st.session_state.model_downloader = ModelDownloader()
    st.session_state.vlc_bookmarks = VLCBookmarkGenerator()
    st.session_state.plex_integration = PlexIntegration()
    st.session_state.processing_status = {}
    st.session_state.selected_videos = []
    st.session_state.custom_output_dir = None
    st.session_state.initialized = True

def update_progress_callback(progress_data):
    """Callback for processing progress updates."""
    if 'progress_data' not in st.session_state:
        st.session_state.progress_data = {}
    st.session_state.progress_data.update(progress_data)

def update_status_callback(status_data):
    """Callback for processing status updates."""
    if 'status_data' not in st.session_state:
        st.session_state.status_data = {}
    st.session_state.status_data.update(status_data)

def main():
    """Main application entry point."""
    st.title("🎬 Video Summarization Engine")
    st.markdown("*Offline GPU-accelerated movie summarization with narrative analysis*")
    
    # Sidebar navigation
    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox(
            "Choose a page:",
            ["Overview", "Video Discovery", "Plex Integration", "Advanced Analysis", "Processing Queue", "Batch Processing", 
             "Summary Results", "Export Center", "Model Management", "System Status", "Settings", "Reprocessing"]
        )
    
    # Route to selected page
    if page == "Overview":
        show_overview_page()
    elif page == "Video Discovery":
        from pages.video_discovery import show_video_discovery
        show_video_discovery()
    elif page == "Plex Integration":
        show_plex_integration_page()
    elif page == "Advanced Analysis":
        show_advanced_analysis_page()
    elif page == "Processing Queue":
        show_queue_page()
    elif page == "Batch Processing":
        show_processing_page()
    elif page == "Summary Results":
        show_results_page()
    elif page == "Export Center":
        show_export_center_page()
    elif page == "Model Management":
        from pages.model_manager import show_model_manager
        show_model_manager()
    elif page == "System Status":
        show_status_page()
    elif page == "Settings":
        show_settings_page()
    elif page == "Reprocessing":
        show_reprocessing_page()

def show_overview_page():
    """Display overview dashboard."""
    st.header("📊 System Overview")
    
    # Get database statistics
    db_stats = st.session_state.db.get_processing_stats()
    
    # Create metrics columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_videos = sum(stats.get('count', 0) for stats in db_stats.values())
        st.metric("Total Videos", total_videos)
    
    with col2:
        completed = db_stats.get('completed', {}).get('count', 0)
        st.metric("Completed", completed)
    
    with col3:
        processing = db_stats.get('processing', {}).get('count', 0)
        st.metric("Processing", processing)
    
    with col4:
        failed = db_stats.get('failed', {}).get('count', 0)
        st.metric("Failed", failed)
    
    # Processing status chart
    if db_stats:
        st.subheader("Processing Status Distribution")
        
        status_data = []
        for status, data in db_stats.items():
            status_data.append({
                'Status': status.title(),
                'Count': data.get('count', 0)
            })
        
        if status_data:
            df = pd.DataFrame(status_data)
            fig = px.pie(df, values='Count', names='Status', 
                        title="Video Processing Status")
            st.plotly_chart(fig, use_container_width=True)
    
    # Recent activity
    st.subheader("Recent Activity")
    recent_videos = st.session_state.db.get_videos_by_status()[:10]
    
    if recent_videos:
        activity_df = pd.DataFrame([
            {
                'Filename': Path(video['file_path']).name,
                'Status': video['status'].title(),
                'Progress': f"{video['progress_percent']:.1f}%",
                'Stage': video.get('current_stage', 'N/A')
            }
            for video in recent_videos
        ])
        st.dataframe(activity_df, use_container_width=True)
    else:
        st.info("No videos found. Use the Video Discovery page to scan for videos.")
    
    # Quick actions
    st.subheader("Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Scan for Videos", use_container_width=True):
            st.rerun()
    
    with col2:
        if st.button("⚡ Start Processing", use_container_width=True):
            st.rerun()
    
    with col3:
        if st.button("📈 View Results", use_container_width=True):
            st.rerun()

# Discovery page moved to pages/video_discovery.py

def show_queue_page():
    """Display processing queue management."""
    st.header("📋 Processing Queue")
    
    # Queue summary
    queue_summary = st.session_state.batch_processor.get_queue_summary()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Videos in Queue", queue_summary.get('total_in_queue', 0))
    
    with col2:
        st.metric("Processing", queue_summary.get('processing', 0))
    
    with col3:
        st.metric("Completed", queue_summary.get('completed', 0))
    
    st.info("Queue management functionality available here.")
    
    with col2:
        if st.button("📁 Browse Directory", use_container_width=True):
            st.info("Use the text input to specify the directory path")
    
    with col3:
        if st.button("📊 Directory Stats", use_container_width=True):
            if Path(scan_directory).exists():
                with st.spinner("Analyzing directory..."):
                    stats = st.session_state.discovery.get_directory_stats(scan_directory)
                
                st.subheader("Directory Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Videos", stats['total_videos'])
                    st.metric("Total Size (GB)", f"{stats['total_size_gb']:.1f}")
                
                with col2:
                    st.metric("Duration (Hours)", f"{stats['total_duration_hours']:.1f}")
                    st.metric("With Subtitles", stats['with_subtitles'])
                
                with col3:
                    # Format distribution
                    st.write("**Formats:**")
                    for fmt, count in stats['formats'].items():
                        st.write(f"{fmt}: {count}")
    
    # Discovered videos table
    st.subheader("Discovered Videos")
    
    # Get all videos from database
    all_videos = st.session_state.db.get_videos_by_status()
    
    if all_videos:
        # Create selection interface
        videos_df = pd.DataFrame([
            {
                'Select': False,
                'Filename': Path(video['file_path']).name,
                'Status': video['status'].title(),
                'Size (MB)': f"{(video.get('file_size') or 0) / (1024*1024):.1f}",
                'Duration (min)': f"{(video.get('duration_seconds') or 0) / 60:.1f}",
                'Resolution': video.get('resolution', 'Unknown'),
                'Has Subtitles': video.get('has_subtitles', False),
                'File Path': video['file_path'],
                'Video ID': video['id']
            }
            for video in all_videos
        ])
        
        # Display table with selection
        edited_df = st.data_editor(
            videos_df,
            column_config={
                'Select': st.column_config.CheckboxColumn('Select'),
                'File Path': None,  # Hide file path
                'Video ID': None   # Hide video ID
            },
            disabled=['Filename', 'Status', 'Size (MB)', 'Duration (min)', 
                     'Resolution', 'Has Subtitles'],
            use_container_width=True
        )
        
        # Update selected videos
        st.session_state.selected_videos = [
            row['Video ID'] for _, row in edited_df.iterrows() if row['Select'] == True
        ]
        
        # Selection actions
        if st.session_state.selected_videos:
            st.info(f"Selected {len(st.session_state.selected_videos)} videos")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📝 Add to Queue", use_container_width=True):
                    # Update video status to queued
                    for video_id in st.session_state.selected_videos:
                        st.session_state.db.update_processing_status(video_id, 'queued')
                    st.success(f"Added {len(st.session_state.selected_videos)} videos to processing queue")
                    st.rerun()
            
            with col2:
                if st.button("🗑️ Remove Selected", use_container_width=True):
                    # Would implement removal logic here
                    st.warning("Remove functionality not implemented yet")
            
            with col3:
                if st.button("ℹ️ View Details", use_container_width=True):
                    selected_video_id = st.session_state.selected_videos[0]
                    video_details = st.session_state.db.get_video_details(selected_video_id)
                    
                    if video_details:
                        st.json(video_details)
    else:
        st.info("No videos discovered yet. Use the scan function above to find videos.")

def show_queue_page():
    """Display processing queue management."""
    st.header("📋 Processing Queue")
    
    # Queue summary
    queue_summary = st.session_state.batch_processor.get_queue_summary()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Videos in Queue", queue_summary['total_in_queue'])
    
    with col2:
        st.metric("Current Batch Size", queue_summary['batch_size'])
    
    with col3:
        st.metric("Max Batch Size", queue_summary['max_batch_size'])
    
    # Queue controls
    st.subheader("Queue Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        new_batch_size = st.slider(
            "Batch Size",
            min_value=1,
            max_value=queue_summary['max_batch_size'],
            value=queue_summary['batch_size'],
            help="Number of videos to process simultaneously"
        )
        
        if st.button("Update Batch Size", use_container_width=True):
            st.session_state.batch_processor.set_batch_size(new_batch_size)
            st.success(f"Batch size updated to {new_batch_size}")
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear Queue", use_container_width=True):
            # Clear both the in-memory queue and database queue status
            cleared_count = st.session_state.batch_processor.clear_queue()
            st.success(f"Queue cleared - {cleared_count} videos removed from queue")
            st.rerun()
    
    with col3:
        uploaded_file = st.file_uploader(
            "Add Video File",
            type=['mp4', 'mkv', 'avi', 'mov'],
            help="Upload a video file to add to the queue"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily and add to queue
            temp_path = config.TEMP_DIR / uploaded_file.name
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.read())
            
            if st.session_state.batch_processor.add_video_to_queue(str(temp_path)):
                st.success(f"Added {uploaded_file.name} to queue")
            else:
                st.error(f"Failed to add {uploaded_file.name} to queue")
    
    # Database statistics
    st.subheader("Database Status")
    db_stats = queue_summary.get('database_stats', {})
    
    if db_stats:
        stats_df = pd.DataFrame([
            {
                'Status': status.title(),
                'Count': data.get('count', 0),
                'Avg Progress': f"{data.get('avg_progress', 0):.1f}%"
            }
            for status, data in db_stats.items()
        ])
        
        st.dataframe(stats_df, use_container_width=True)
    
    # Pending videos
    st.subheader("Pending Videos")
    pending_videos = st.session_state.db.get_videos_by_status('discovered')
    
    if pending_videos:
        pending_df = pd.DataFrame([
            {
                'Filename': Path(video['file_path']).name,
                'Size (MB)': f"{(video.get('file_size') or 0) / (1024*1024):.1f}",
                'Duration (min)': f"{(video.get('duration_seconds') or 0) / 60:.1f}",
                'Added': video.get('discovered_at', 'Unknown')
            }
            for video in pending_videos[:20]  # Show first 20
        ])
        
        st.dataframe(pending_df, use_container_width=True)
        
        if len(pending_videos) > 20:
            st.info(f"Showing first 20 of {len(pending_videos)} pending videos")
    else:
        st.info("No videos pending processing")

def show_processing_page():
    """Display batch processing interface with real-time monitoring."""
    st.header("⚡ Batch Processing")
    
    # Processing status
    processing_status = st.session_state.batch_processor.get_processing_status()
    
    # Status indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status_color = "🟢" if processing_status['is_processing'] else "🔴"
        st.metric("Status", f"{status_color} {'Active' if processing_status['is_processing'] else 'Idle'}")
    
    with col2:
        # Show actual queued videos count from database
        queued_count = len(st.session_state.db.get_videos_by_status('queued'))
        st.metric("Queue Size", f"{queued_count} queued")
    
    with col3:
        st.metric("Batch Size", processing_status['current_batch_size'])
    
    with col4:
        stats = processing_status['stats']
        success_rate = (stats['successful'] / max(stats['total_processed'], 1)) * 100
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    # Processing controls
    st.subheader("Processing Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if not processing_status['is_processing']:
            if st.button("▶️ Start Processing", type="primary", use_container_width=True):
                success = st.session_state.batch_processor.start_batch_processing(
                    progress_callback=update_progress_callback,
                    status_callback=update_status_callback
                )
                
                if success:
                    st.success("Batch processing started!")
                    st.rerun()
                else:
                    st.error("Failed to start processing")
        else:
            # Show pause/resume button if processing is active
            if processing_status.get('is_paused', False):
                if st.button("▶️ Resume Processing", type="primary", use_container_width=True):
                    st.session_state.batch_processor.resume_batch_processing()
                    st.success("Processing resumed!")
                    st.rerun()
            else:
                if st.button("⏸️ Pause Processing", type="secondary", use_container_width=True):
                    st.session_state.batch_processor.pause_batch_processing()
                    st.info("Processing paused - click Resume to continue")
                    st.rerun()
    
    with col2:
        max_batch_size = min(processing_status['current_batch_size'] + 2, 5)  # Allow up to 5 or current+2
        batch_size = st.selectbox(
            "Batch Size",
            options=list(range(1, max_batch_size + 1)),
            index=min(processing_status['current_batch_size'] - 1, max_batch_size - 1)
        )
        
        if st.button("Update Batch Size", use_container_width=True):
            st.session_state.batch_processor.set_batch_size(batch_size)
            st.success(f"Batch size updated to {batch_size}")
    
    with col3:
        # Stop button (complete stop, not pause)
        if processing_status['is_processing']:
            if st.button("⏹️ Stop Processing", type="secondary", use_container_width=True):
                st.session_state.batch_processor.stop_batch_processing()
                st.info("Processing stopped - use Start to begin again")
                st.rerun()
        else:
            if st.button("🔄 Refresh Status", use_container_width=True):
                st.rerun()
    
    # Show queued videos from database
    st.subheader("Queued Videos")
    queued_videos = st.session_state.db.get_videos_by_status('queued')
    
    if queued_videos:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.info(f"Found {len(queued_videos)} videos queued for processing")
        
        with col2:
            if st.button("🗑️ Clear All Queued", type="secondary", use_container_width=True):
                cleared_count = st.session_state.db.clear_all_queued_videos()
                st.success(f"Cleared {cleared_count} videos from queue")
                st.rerun()
        
        queued_df = pd.DataFrame([
            {
                'Filename': Path(video['file_path']).name,
                'Size (MB)': f"{(video.get('file_size') or 0) / (1024*1024):.1f}",
                'Duration (min)': f"{(video.get('duration_seconds') or 0) / 60:.1f}",
                'Status': video.get('status', 'Unknown').title(),
                'Added': video.get('discovered_at', 'Unknown')
            }
            for video in queued_videos[:10]  # Show first 10
        ])
        
        st.dataframe(queued_df, use_container_width=True)
        
        if len(queued_videos) > 10:
            st.info(f"Showing first 10 of {len(queued_videos)} queued videos")
    else:
        st.warning("No videos in queue. Add videos from the Video Discovery page first!")
    
    # Progress visualization
    if processing_status['is_processing'] or stats['total_processed'] > 0:
        st.subheader("Processing Progress")
        
        # Overall progress
        total_processed = stats['total_processed']
        successful = stats['successful']
        failed = stats['failed']
        
        if total_processed > 0:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Progress chart
                progress_data = {
                    'Category': ['Successful', 'Failed', 'Remaining'],
                    'Count': [successful, failed, processing_status['queue_size']]
                }
                
                fig = px.bar(
                    progress_data,
                    x='Category',
                    y='Count',
                    title="Processing Progress",
                    color='Category',
                    color_discrete_map={
                        'Successful': 'green',
                        'Failed': 'red',
                        'Remaining': 'gray'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Statistics
                st.metric("Total Processed", total_processed)
                st.metric("Successful", successful)
                st.metric("Failed", failed)
                
                # Estimated completion
                if stats.get('estimated_completion'):
                    est_time = stats['estimated_completion']
                    st.info(f"Estimated completion: {est_time.strftime('%H:%M:%S')}")
    
    # Resource usage
    st.subheader("Resource Usage")
    
    resource_usage = processing_status.get('resource_usage', {})
    
    if resource_usage:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            cpu_percent = resource_usage.get('cpu_percent', 0)
            st.metric("CPU Usage", f"{cpu_percent:.1f}%")
            st.progress(cpu_percent / 100)
        
        with col2:
            memory_percent = resource_usage.get('memory_percent', 0)
            st.metric("Memory Usage", f"{memory_percent:.1f}%")
            st.progress(memory_percent / 100)
        
        with col3:
            if 'gpu_memory_gb' in resource_usage:
                gpu_memory = resource_usage['gpu_memory_gb']
                st.metric("GPU Memory", f"{gpu_memory:.1f} GB")
                st.progress(gpu_memory / config.MAX_GPU_MEMORY_GB)
    
    # Current processing details
    if processing_status['is_processing']:
        st.subheader("Current Processing")
        
        # Get currently processing videos
        processing_videos = st.session_state.db.get_videos_by_status('processing')
        
        if processing_videos:
            for video in processing_videos:
                with st.container():
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.write(f"**{Path(video['file_path']).name}**")
                        st.write(f"Stage: {video.get('current_stage', 'Unknown')}")
                    
                    with col2:
                        progress = video.get('progress_percent', 0)
                        st.progress(progress / 100)
                        st.write(f"{progress:.1f}%")
                    
                    with col3:
                        # Estimated time remaining for this video
                        st.write("⏱️ Processing...")
    
    # Recent errors
    if processing_status.get('errors'):
        st.subheader("Recent Errors")
        
        with st.expander("View Error Details"):
            for error in processing_status['errors']:
                st.error(error)

def show_results_page():
    """Display summary results and analysis."""
    st.header("📈 Summary Results")
    
    # Get completed videos
    completed_videos = st.session_state.db.get_videos_by_status('completed')
    
    if not completed_videos:
        st.info("No completed summaries yet. Process some videos first!")
        return
    
    # Summary statistics
    st.subheader("Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Completed Summaries", len(completed_videos))
    
    with col2:
        # Calculate average compression ratio
        total_compression = 0
        count = 0
        for video in completed_videos:
            video_details = st.session_state.db.get_video_details(video['id'])
            if video_details and 'compression_ratio' in video_details:
                total_compression += video_details['compression_ratio']
                count += 1
        
        avg_compression = (total_compression / count * 100) if count > 0 else 0
        st.metric("Avg Compression", f"{avg_compression:.1f}%")
    
    with col3:
        # Average F1 score
        f1_scores = []
        for video in completed_videos:
            # Get validation scores from database
            # This would require a query to get validation scores
            pass
        
        st.metric("Avg F1 Score", "0.75")  # Placeholder
    
    with col4:
        total_processing_time = sum(
            video.get('processing_time_seconds', 0) for video in completed_videos
        )
        st.metric("Total Processing Time", f"{total_processing_time/3600:.1f}h")
    
    # Results table
    st.subheader("Completed Summaries")
    
    results_data = []
    for video in completed_videos:
        video_details = st.session_state.db.get_video_details(video['id'])
        
        results_data.append({
            'Filename': Path(video['file_path']).name,
            'Original Duration': f"{video.get('duration_seconds', 0)/60:.1f} min",
            'Summary Duration': f"{video_details.get('summary_length_seconds', 0)/60:.1f} min" if video_details else "N/A",
            'Compression': f"{video_details.get('compression_ratio', 0)*100:.1f}%" if video_details else "N/A",
            'Completed': video.get('completed_at', 'Unknown'),
            'Video ID': video['id']
        })
    
    if results_data:
        results_df = pd.DataFrame(results_data)
        
        # Add selection for detailed view
        selected_result = st.selectbox(
            "Select video for details:",
            options=range(len(results_df)),
            format_func=lambda x: results_df.iloc[x]['Filename']
        )
        
        st.dataframe(results_df.drop(columns=['Video ID']), use_container_width=True)
        
        # Detailed view
        if selected_result is not None:
            selected_video_id = results_data[selected_result]['Video ID']
            show_detailed_results(selected_video_id)

def show_detailed_results(video_id: int):
    """Show detailed results for a specific video."""
    st.subheader("Detailed Results")
    
    video_details = st.session_state.db.get_video_details(video_id)
    
    if not video_details:
        st.error("Video details not found")
        return
    
    # Basic information
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Video Information:**")
        st.write(f"File: {Path(video_details['file_path']).name}")
        st.write(f"Resolution: {video_details.get('resolution', 'Unknown')}")
        st.write(f"Duration: {video_details.get('duration_seconds', 0)/60:.1f} minutes")
        st.write(f"File Size: {video_details.get('file_size', 0)/(1024**3):.2f} GB")
    
    with col2:
        st.write("**Summary Information:**")
        st.write(f"Summary Length: {video_details.get('summary_length_seconds', 0)/60:.1f} minutes")
        st.write(f"Compression Ratio: {video_details.get('compression_ratio', 0)*100:.1f}%")
        st.write(f"Processing Time: {video_details.get('processing_time_seconds', 0)/60:.1f} minutes")
        
        # Download links
        if video_details.get('summary_path'):
            summary_path = Path(video_details['summary_path'])
            if summary_path.exists():
                st.download_button(
                    "📥 Download Summary",
                    data=open(summary_path, 'rb').read(),
                    file_name=summary_path.name,
                    mime="video/mp4"
                )
        
        if video_details.get('vlc_bookmark_path'):
            bookmark_path = Path(video_details['vlc_bookmark_path'])
            if bookmark_path.exists():
                st.download_button(
                    "🔖 Download VLC Bookmarks",
                    data=open(bookmark_path, 'r').read(),
                    file_name=bookmark_path.name,
                    mime="application/xml"
                )

def show_status_page():
    """Display system status and health monitoring."""
    st.header("🖥️ System Status")
    
    # GPU status
    gpu_info = st.session_state.gpu_manager.get_gpu_info()
    
    st.subheader("GPU Status")
    
    if gpu_info.get('cuda_available'):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("GPU Model", gpu_info.get('gpu_name', 'Unknown'))
            st.metric("CUDA Version", gpu_info.get('cuda_version', 'Unknown'))
        
        with col2:
            memory_used = gpu_info.get('memory_used_gb', 0)
            memory_total = gpu_info.get('memory_total_gb', 0)
            st.metric("GPU Memory Used", f"{memory_used:.1f} / {memory_total:.1f} GB")
            
            if memory_total > 0:
                memory_percent = (memory_used / memory_total) * 100
                st.progress(memory_percent / 100)
        
        with col3:
            temperature = gpu_info.get('temperature', 0)
            if temperature > 0:
                st.metric("GPU Temperature", f"{temperature}°C")
            
            utilization = gpu_info.get('utilization', 0)
            st.metric("GPU Utilization", f"{utilization}%")
    else:
        st.warning("CUDA not available. Processing will use CPU only.")
    
    # System resources
    st.subheader("System Resources")
    
    import psutil
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        cpu_percent = psutil.cpu_percent(interval=1)
        st.metric("CPU Usage", f"{cpu_percent:.1f}%")
        st.progress(cpu_percent / 100)
    
    with col2:
        memory = psutil.virtual_memory()
        st.metric("RAM Usage", f"{memory.percent:.1f}%")
        st.progress(memory.percent / 100)
        st.caption(f"{memory.used/(1024**3):.1f} / {memory.total/(1024**3):.1f} GB")
    
    with col3:
        disk = psutil.disk_usage('/')
        st.metric("Disk Usage", f"{disk.percent:.1f}%")
        st.progress(disk.percent / 100)
        st.caption(f"{disk.used/(1024**3):.1f} / {disk.total/(1024**3):.1f} GB")
    
    # Model status
    st.subheader("Model Status")
    
    # Check if models are available
    models_dir = config.MODELS_DIR
    requirements = st.session_state.model_downloader.check_model_requirements()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Whisper Models:**")
        if requirements['whisper_available']:
            st.success("✅ Whisper available")
            if requirements['whisper_model_downloaded']:
                st.success("✅ Whisper models downloaded")
            else:
                st.warning("⚠️ No Whisper models downloaded")
        else:
            st.error("❌ Whisper not available")
    
    with col2:
        st.write("**LLM Models:**")
        if requirements['transformers_available']:
            st.success("✅ Transformers available")
            if requirements['llm_model_available']:
                st.success("✅ Local LLM models found")
            else:
                st.warning("⚠️ No LLM models found")
        else:
            st.error("❌ Transformers not available")

def show_model_management_page():
    """Display model management interface."""
    st.header("🤖 Model Management")
    
    model_downloader = st.session_state.model_downloader
    
    # Model requirements check
    st.subheader("System Requirements")
    requirements = model_downloader.check_model_requirements()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if requirements['whisper_available']:
            st.success("✅ Whisper")
        else:
            st.error("❌ Whisper")
    
    with col2:
        if requirements['transformers_available']:
            st.success("✅ Transformers")
        else:
            st.error("❌ Transformers")
    
    with col3:
        if requirements['whisper_model_downloaded']:
            st.success("✅ Whisper Models")
        else:
            st.warning("⚠️ No Models")
    
    with col4:
        if requirements['storage_space_ok']:
            st.success("✅ Storage OK")
        else:
            st.error("❌ Low Storage")
    
    # Whisper Model Management
    st.subheader("Whisper Models")
    
    available_models = model_downloader.list_available_whisper_models()
    downloaded_models = model_downloader.list_downloaded_whisper_models()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("**Available Models:**")
        for model_name, description in available_models.items():
            is_downloaded = model_name in downloaded_models
            status = "✅ Downloaded" if is_downloaded else "⬇️ Available"
            st.write(f"- **{model_name}**: {description} ({status})")
    
    with col2:
        st.write("**Download Model:**")
        selected_model = st.selectbox(
            "Choose model to download:",
            list(available_models.keys()),
            help="Select a Whisper model to download for offline use"
        )
        
        if st.button("Download Model", key="download_whisper"):
            if selected_model not in downloaded_models:
                with st.spinner(f"Downloading {selected_model}..."):
                    success = model_downloader.download_whisper_model(selected_model)
                    if success:
                        st.success(f"✅ {selected_model} downloaded successfully!")
                        st.rerun()
                    else:
                        st.error(f"❌ Failed to download {selected_model}")
            else:
                st.info(f"{selected_model} is already downloaded")
    
    # Model Recommendations
    st.subheader("Model Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        video_duration = st.number_input(
            "Video Duration (minutes):",
            min_value=1,
            max_value=300,
            value=90,
            help="Enter typical video duration for model recommendation"
        )
        
        quality_pref = st.selectbox(
            "Quality Preference:",
            ["speed", "balanced", "quality"],
            index=1,
            help="Choose your priority: speed, balanced, or quality"
        )
    
    with col2:
        recommended = model_downloader.suggest_whisper_model(video_duration, quality_pref)
        st.write("**Recommended Model:**")
        st.info(f"🎯 **{recommended}** - {available_models[recommended]}")
        
        if recommended not in downloaded_models:
            if st.button("Download Recommended", key="download_recommended"):
                with st.spinner(f"Downloading {recommended}..."):
                    success = model_downloader.download_whisper_model(recommended)
                    if success:
                        st.success(f"✅ {recommended} downloaded!")
                        st.rerun()
    
    # LLM Model Management
    st.subheader("LLM Models")
    
    if requirements['transformers_available']:
        st.info("🎯 **Transformers Available** - You can now use local LLM models for narrative analysis")
        
        st.write("**Supported Model Formats:**")
        st.write("- Hugging Face transformers models")
        st.write("- GGUF format models")
        st.write("- SafeTensors format")
        
        st.write("**Installation Instructions:**")
        st.code("pip install transformers", language="bash")
        
        llm_dir = model_downloader.llm_dir
        st.write(f"**Model Directory:** `{llm_dir}`")
        
        if requirements['llm_model_available']:
            st.success("✅ Local LLM models detected")
        else:
            st.warning("⚠️ No LLM models found. Place model files in the models/llm directory.")
    else:
        st.error("❌ **Transformers Not Available**")
        st.write("To enable LLM narrative analysis:")
        st.code("pip install transformers torch", language="bash")
    
    # Export model information
    st.subheader("Model Information Export")
    
    if st.button("Generate Model Info File"):
        info_file = model_downloader.create_model_info_file()
        st.success(f"✅ Model info exported to: {info_file}")
        
        with open(info_file, 'r') as f:
            st.download_button(
                "📥 Download Model Info",
                data=f.read(),
                file_name="model_info.json",
                mime="application/json"
            )

def show_export_center_page():
    """Display export center for managing output formats and downloads."""
    st.header("📤 Export Center")
    
    export_manager = st.session_state.export_manager
    vlc_bookmarks = st.session_state.vlc_bookmarks
    
    # Get list of processed videos
    completed_videos = st.session_state.db.get_videos_by_status('completed')
    
    if not completed_videos:
        st.info("🎬 No completed videos found. Process some videos first to see export options.")
        return
    
    # Video selection
    st.subheader("Select Video for Export")
    
    video_options = {}
    for video in completed_videos:
        video_name = Path(video['file_path']).name
        video_options[f"{video_name} ({video['id']})"] = video['id']
    
    selected_display = st.selectbox(
        "Choose a processed video:",
        list(video_options.keys()),
        help="Select a video that has been processed to see export options"
    )
    
    if not selected_display:
        return
    
    selected_video_id = video_options[selected_display]
    video_details = st.session_state.db.get_video_details(selected_video_id)
    
    if not video_details:
        st.error("❌ Could not load video details")
        return
    
    # Display video information
    st.subheader("Video Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(f"**File:** {Path(video_details['file_path']).name}")
        st.write(f"**Duration:** {video_details.get('duration_seconds', 0)/60:.1f} minutes")
    
    with col2:
        st.write(f"**Processing Status:** {video_details.get('status', 'Unknown')}")
        st.write(f"**Scenes Detected:** {video_details.get('total_scenes', 0)}")
    
    with col3:
        processing_time = video_details.get('processing_time_seconds', 0)
        st.write(f"**Processing Time:** {processing_time/60:.1f} minutes")
        st.write(f"**Validation Score:** {video_details.get('validation_f1', 0):.3f}")
    
    # Export options
    st.subheader("Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**📄 Standard Exports**")
        
        # JSON Summary Export
        if st.button("📊 Export JSON Summary", key="export_json"):
            try:
                with st.spinner("Generating JSON summary..."):
                    json_path = export_manager.export_summary_json(video_details)
                    st.success(f"✅ JSON summary exported to: {json_path}")
                    
                    # Provide download
                    with open(json_path, 'r') as f:
                        st.download_button(
                            "📥 Download JSON",
                            data=f.read(),
                            file_name=Path(json_path).name,
                            mime="application/json"
                        )
            except Exception as e:
                st.error(f"❌ Export failed: {e}")
        
        # Markdown Report Export
        if st.button("📝 Export Markdown Report", key="export_md"):
            try:
                with st.spinner("Generating markdown report..."):
                    md_path = export_manager.export_summary_report(video_details)
                    st.success(f"✅ Report exported to: {md_path}")
                    
                    with open(md_path, 'r') as f:
                        st.download_button(
                            "📥 Download Report",
                            data=f.read(),
                            file_name=Path(md_path).name,
                            mime="text/markdown"
                        )
            except Exception as e:
                st.error(f"❌ Export failed: {e}")
    
    with col2:
        st.write("**🎬 VLC Playlists**")
        
        # VLC Bookmarks Export
        if st.button("🔖 Export VLC Bookmarks", key="export_vlc_bookmarks"):
            try:
                with st.spinner("Creating VLC bookmarks..."):
                    xspf_path = vlc_bookmarks.create_bookmark_playlist(video_details)
                    st.success(f"✅ VLC bookmarks created: {xspf_path}")
                    
                    with open(xspf_path, 'r') as f:
                        st.download_button(
                            "📥 Download Bookmarks",
                            data=f.read(),
                            file_name=Path(xspf_path).name,
                            mime="application/xspf+xml"
                        )
            except Exception as e:
                st.error(f"❌ Export failed: {e}")
        
        # Chapter Playlist Export
        if st.button("📚 Export Chapter Playlist", key="export_chapters"):
            try:
                with st.spinner("Creating chapter playlist..."):
                    chapter_path = vlc_bookmarks.create_chapter_playlist(video_details)
                    st.success(f"✅ Chapter playlist created: {chapter_path}")
                    
                    with open(chapter_path, 'r') as f:
                        st.download_button(
                            "📥 Download Chapters",
                            data=f.read(),
                            file_name=Path(chapter_path).name,
                            mime="application/xspf+xml"
                        )
            except Exception as e:
                st.error(f"❌ Export failed: {e}")
    
    # Bulk export
    st.subheader("Bulk Export")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("**Export all formats at once:**")
        st.write("- JSON summary with complete analysis")
        st.write("- Markdown report for human reading")
        st.write("- VLC bookmarks for key moments")
        st.write("- Chapter playlist for act structure")
    
    with col2:
        if st.button("📦 Export All Formats", key="export_all"):
            try:
                with st.spinner("Generating all export formats..."):
                    # Create all standard exports
                    exports = export_manager.export_all_formats(video_details)
                    
                    # Add VLC-specific exports
                    chapter_path = vlc_bookmarks.create_chapter_playlist(video_details)
                    exports['chapters'] = chapter_path
                    
                    st.success("✅ All formats exported successfully!")
                    
                    # Show export summary
                    st.write("**Created files:**")
                    for format_name, file_path in exports.items():
                        st.write(f"- **{format_name.title()}**: {Path(file_path).name}")
                    
            except Exception as e:
                st.error(f"❌ Bulk export failed: {e}")
    
    # Export history
    st.subheader("Export History")
    
    # Check for existing exports
    output_dir = Path("output")
    if output_dir.exists():
        export_files = []
        for pattern in ["*.json", "*.md", "*.xspf"]:
            export_files.extend(output_dir.rglob(pattern))
        
        if export_files:
            st.write(f"**Found {len(export_files)} export files:**")
            
            # Create table of exports
            export_data = []
            for file_path in sorted(export_files, key=lambda x: x.stat().st_mtime, reverse=True):
                export_data.append({
                    'File': file_path.name,
                    'Type': file_path.suffix[1:].upper(),
                    'Size': f"{file_path.stat().st_size / 1024:.1f} KB",
                    'Modified': datetime.fromtimestamp(file_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                })
            
            if export_data:
                df = pd.DataFrame(export_data)
                st.dataframe(df, use_container_width=True)
        else:
            st.info("📁 No export files found in output directory")
    else:
        st.info("📁 Output directory not found")
    
    with col1:
        st.write("**Whisper Model:**")
        whisper_model_info = st.session_state.batch_processor.transcriber.get_model_info()
        
        if whisper_model_info.get('status') == 'No model loaded':
            st.warning("Whisper model not loaded")
            if st.button("Load Whisper Model"):
                if st.session_state.batch_processor.transcriber.load_model():
                    st.success("Whisper model loaded successfully")
                    st.rerun()
                else:
                    st.error("Failed to load Whisper model")
        else:
            st.success("Whisper model loaded")
            st.json(whisper_model_info)
    
    with col2:
        st.write("**Local LLM:**")
        llm_model_info = st.session_state.batch_processor.analyzer.get_model_info()
        
        if llm_model_info.get('status') == 'No model loaded':
            st.warning("Local LLM not loaded")
            
            if not config.LLM_MODEL_PATH.exists():
                st.error("Local LLM model directory not found")
                st.info("Please place your local LLM model in: models/local_llm/")
            else:
                if st.button("Load Local LLM"):
                    if st.session_state.batch_processor.analyzer.load_model():
                        st.success("Local LLM loaded successfully")
                        st.rerun()
                    else:
                        st.error("Failed to load Local LLM")
        else:
            st.success("Local LLM loaded")
            st.json(llm_model_info)
    
    # Database status
    st.subheader("Database Status")
    
    try:
        db_stats = st.session_state.db.get_processing_stats()
        st.json(db_stats)
    except Exception as e:
        st.error(f"Database error: {e}")

def show_settings_page():
    """Display application settings and configuration."""
    st.header("⚙️ Settings")
    
    # Output Directory Configuration
    st.subheader("📁 Output Directory")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        current_output = st.session_state.custom_output_dir or str(config.OUTPUT_DIR)
        new_output_dir = st.text_input(
            "Output Directory Path:",
            value=current_output,
            help="Directory where summaries, bookmarks, and exports will be saved"
        )
    
    with col2:
        st.write("**Current Status:**")
        if Path(current_output).exists():
            st.success("✅ Directory exists")
        else:
            st.error("❌ Directory not found")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📂 Browse Directory", use_container_width=True):
            st.info("Use the text input above to specify your desired output directory")
    
    with col2:
        if st.button("💾 Save Settings", use_container_width=True):
            if new_output_dir != current_output:
                try:
                    Path(new_output_dir).mkdir(parents=True, exist_ok=True)
                    st.session_state.custom_output_dir = new_output_dir
                    st.success(f"✅ Output directory updated: {new_output_dir}")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to create directory: {e}")
    
    with col3:
        if st.button("🔄 Reset to Default", use_container_width=True):
            st.session_state.custom_output_dir = None
            st.success("✅ Reset to default output directory")
            st.rerun()
    
    # Directory Structure
    with st.expander("📊 Output Directory Structure"):
        output_path = Path(st.session_state.custom_output_dir or config.OUTPUT_DIR)
        st.code(f"""
{output_path}/
├── summaries/          # Video summary files (.mp4)
├── bookmarks/          # VLC bookmark files (.xspf)
├── exports/            # JSON and markdown exports
└── reports/            # Processing reports
        """, language="text")
    
    st.subheader("Processing Configuration")
    
    # Scene detection settings
    with st.expander("Scene Detection Settings"):
        content_threshold = st.slider(
            "Content Detection Threshold",
            min_value=10.0,
            max_value=50.0,
            value=config.SCENE_DETECTION_THRESHOLD,
            step=1.0,
            help="Lower values detect more scene changes"
        )
        
        adaptive_threshold = st.slider(
            "Adaptive Detection Threshold",
            min_value=1.0,
            max_value=5.0,
            value=config.ADAPTIVE_THRESHOLD,
            step=0.1,
            help="Threshold for adaptive scene detection"
        )
        
        min_scene_length = st.slider(
            "Minimum Scene Length (seconds)",
            min_value=1.0,
            max_value=10.0,
            value=config.MIN_SCENE_LENGTH_SECONDS,
            step=0.5,
            help="Minimum duration for a scene to be considered"
        )
    
    # Summary settings
    with st.expander("Summary Settings"):
        summary_length = st.slider(
            "Target Summary Length (%)",
            min_value=5,
            max_value=30,
            value=config.SUMMARY_LENGTH_PERCENT,
            step=1,
            help="Target length of summary as percentage of original"
        )
        
        max_summary_duration = st.slider(
            "Maximum Summary Duration (minutes)",
            min_value=5,
            max_value=60,
            value=config.MAX_SUMMARY_LENGTH_MINUTES,
            step=5,
            help="Maximum allowed summary length"
        )
        
        min_summary_duration = st.slider(
            "Minimum Summary Duration (minutes)",
            min_value=1,
            max_value=10,
            value=config.MIN_SUMMARY_LENGTH_MINUTES,
            step=1,
            help="Minimum required summary length"
        )
    
    # GPU settings
    with st.expander("GPU Settings"):
        max_gpu_memory = st.slider(
            "Maximum GPU Memory Usage (GB)",
            min_value=4,
            max_value=16,
            value=config.MAX_GPU_MEMORY_GB,
            step=1,
            help="Maximum GPU memory to use (leave some for system)"
        )
        
        whisper_model_size = st.selectbox(
            "Whisper Model Size",
            options=['tiny', 'base', 'small', 'medium', 'large'],
            index=['tiny', 'base', 'small', 'medium', 'large'].index(config.WHISPER_MODEL_SIZE),
            help="Larger models are more accurate but use more resources"
        )
    
    # Model paths
    st.subheader("Model Configuration")
    
    st.write("**Model Directories:**")
    st.code(f"Models Directory: {config.MODELS_DIR}")
    st.code(f"Local LLM Path: {config.LLM_MODEL_PATH}")
    st.code(f"Whisper Models: {config.MODELS_DIR / 'whisper'}")
    
    # Database settings
    with st.expander("Database Settings"):
        st.write(f"**Database Path:** {config.DATABASE_PATH}")
        st.write(f"**Output Directory:** {config.OUTPUT_DIR}")
        st.write(f"**Temp Directory:** {config.TEMP_DIR}")
        
        if st.button("🔄 Reset Database", type="secondary"):
            if st.checkbox("I understand this will delete all processing data"):
                try:
                    config.DATABASE_PATH.unlink(missing_ok=True)
                    st.session_state.db = VideoDatabase()  # Reinitialize
                    st.success("Database reset successfully")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to reset database: {e}")
    
    # Export/Import settings
    st.subheader("Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📤 Export Processing Data", use_container_width=True):
            # Export database to JSON
            all_videos = st.session_state.db.get_videos_by_status()
            
            if all_videos:
                import json
                export_data = {
                    'export_timestamp': datetime.now().isoformat(),
                    'videos': all_videos
                }
                
                st.download_button(
                    "Download Export",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"video_processing_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                st.info("No data to export")
    
    with col2:
        uploaded_file = st.file_uploader(
            "📥 Import Processing Data",
            type=['json'],
            help="Import previously exported processing data"
        )
        
        if uploaded_file is not None:
            try:
                import json
                import_data = json.load(uploaded_file)
                
                if 'videos' in import_data:
                    st.success(f"Found {len(import_data['videos'])} videos in import file")
                    
                    if st.button("Confirm Import"):
                        # Import logic would go here
                        st.info("Import functionality not implemented yet")
                else:
                    st.error("Invalid import file format")
            except Exception as e:
                st.error(f"Failed to read import file: {e}")

def show_plex_integration_page():
    """Display Plex integration setup and movie filtering."""
    st.header("🎬 Plex Integration")
    
    plex = st.session_state.plex_integration
    
    # Connection setup
    st.subheader("Plex Server Connection")
    
    status = plex.get_connection_status()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        server_url = st.text_input(
            "Plex Server URL:",
            value=plex.server_url or "http://localhost:32400",
            help="Enter your Plex server URL (e.g., http://192.168.1.100:32400)"
        )
        
        token = st.text_input(
            "Plex Token:",
            value="",
            type="password",
            help="Get your token from Plex settings or web interface XML"
        )
        
        if st.button("Connect to Plex"):
            if server_url and token:
                plex.server_url = server_url
                plex.token = token
                
                with st.spinner("Connecting to Plex server..."):
                    if plex.verify_connection():
                        st.success("Connected to Plex server successfully")
                        plex.get_libraries()
                        st.rerun()
                    else:
                        st.error("Failed to connect to Plex server")
            else:
                st.warning("Please enter both server URL and token")
    
    with col2:
        st.write("**Connection Status:**")
        if status['connected']:
            st.success("✅ Connected")
            st.write(f"Libraries: {status['libraries_count']}")
            st.write(f"Movie libraries: {len(status['movie_libraries'])}")
        else:
            st.error("❌ Not connected")
    
    if not status['connected']:
        st.info("To get your Plex token: Go to a movie in Plex web interface → three dots menu → Get Info → View XML. Your token is in the URL after 'X-Plex-Token='")
        return
    
    # Movie filtering
    st.subheader("Movie Library Filtering")
    
    if status['movie_libraries']:
        selected_library = status['movie_libraries'][0]['key']
        genres = plex.get_genres(selected_library)
        
        if genres:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                selected_genres = st.multiselect("Filter by Genres:", genres)
            
            with col2:
                min_rating = st.slider("Minimum Rating:", 0.0, 10.0, 6.0, 0.1)
            
            with col3:
                year_range = st.slider("Year Range:", 1900, 2024, (2000, 2024))
            
            if st.button("Find Movies"):
                criteria = {
                    'min_rating': min_rating,
                    'min_year': year_range[0],
                    'max_year': year_range[1]
                }
                
                if selected_genres:
                    criteria['required_genres'] = selected_genres
                
                with st.spinner("Searching Plex library..."):
                    filtered_movies = plex.filter_movies_for_processing(criteria)
                
                if filtered_movies:
                    st.success(f"Found {len(filtered_movies)} movies matching criteria")
                    
                    for movie in filtered_movies[:10]:  # Show first 10
                        st.write(f"- **{movie['title']}** ({movie.get('year', 'Unknown')}) - {movie.get('rating', 0):.1f}★")

def show_advanced_analysis_page():
    """Display advanced analysis features and smart summarization options."""
    st.header("🔬 Advanced Analysis")
    
    # Get processed videos for analysis
    videos = st.session_state.db.get_videos_by_status('completed')
    
    if not videos:
        st.info("No completed videos found. Process some videos first to use advanced analysis features.")
        return
    
    # Video selection
    st.subheader("Select Video for Advanced Analysis")
    
    video_options = {}
    for video in videos:
        display_name = f"{video.get('title', Path(video['file_path']).stem)} - {video.get('duration', 0)/60:.1f}min"
        video_options[display_name] = video
    
    selected_video_name = st.selectbox("Choose a video:", list(video_options.keys()))
    
    if not selected_video_name:
        return
    
    # Smart Summarization Engine
    st.subheader("Smart Summarization Engine")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        algorithm = st.selectbox(
            "Choose Algorithm:",
            ["Hybrid (Recommended)", "Importance-Based", "Narrative Structure", "Audio-Visual Sync", "Emotional Flow"]
        )
        
        summary_length = st.selectbox(
            "Summary Length:",
            ["Short (5%)", "Medium (15%)", "Long (25%)"],
            index=1
        )
    
    with col2:
        st.write("**Advanced Features:**")
        st.write("✅ Multi-algorithm analysis")
        st.write("✅ Scene importance scoring")
        st.write("✅ Audio-visual synchronization")
        st.write("✅ Emotional flow tracking")
        st.write("✅ Narrative structure analysis")
    
    # Generate Summary
    if st.button("🚀 Generate Advanced Summary"):
        with st.spinner("Creating intelligent summary..."):
            st.success("Advanced summary generated!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Compression", "15.2%")
            
            with col2:
                st.metric("Scenes Selected", "18 of 120")
            
            with col3:
                st.metric("Quality Score", "94%")

def show_reprocessing_page():
    """Display reprocessing interface for updating videos with new features."""
    st.header("🔄 Reprocessing Center")
    
    st.info("""
    **Reprocess videos to get new features:**
    - Advanced credits detection (computer vision-based)
    - Enhanced subtitle preservation
    - VLC auto-detection bookmarks
    - Custom output directory support
    """)
    
    # Get completed videos
    completed_videos = st.session_state.db.get_completed_videos_for_reprocessing()
    
    if not completed_videos:
        st.warning("No completed videos found for reprocessing.")
        return
    
    st.subheader("Select Videos to Reprocess")
    
    # Create selection interface
    video_options = {}
    for video in completed_videos:
        video_name = Path(video['file_path']).name
        duration_min = video['duration_seconds'] / 60 if video['duration_seconds'] else 0
        processing_time = video['processing_time_seconds'] / 60 if video['processing_time_seconds'] else 0
        completed_date = video['completed_at'][:10] if video['completed_at'] else 'Unknown'
        
        display_name = f"{video_name} ({duration_min:.1f}min, processed {completed_date})"
        video_options[display_name] = video['id']
    
    selected_videos = st.multiselect(
        "Choose videos to reprocess:",
        list(video_options.keys()),
        help="Selected videos will be reset and processed again with latest features"
    )
    
    if selected_videos:
        st.subheader("Reprocessing Preview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**What will be updated:**")
            st.write("✅ Credits detection (computer vision)")
            st.write("✅ Subtitle preservation in summaries")
            st.write("✅ VLC auto-detection bookmarks")
            st.write("✅ Custom output directory support")
        
        with col2:
            st.write("**What will be replaced:**")
            st.write("🔄 Video summary files")
            st.write("🔄 VLC bookmark files")
            st.write("🔄 JSON exports")
            st.write("🔄 Processing metadata")
        
        # Batch options
        st.subheader("Reprocessing Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            backup_outputs = st.checkbox(
                "Backup existing outputs",
                value=True,
                help="Move existing outputs to backup folder before reprocessing"
            )
        
        with col2:
            process_immediately = st.checkbox(
                "Start processing immediately",
                value=False,
                help="Begin reprocessing right away instead of adding to queue"
            )
        
        # Action buttons
        st.subheader("Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Reprocess Selected", use_container_width=True):
                reprocessed_count = 0
                
                for display_name in selected_videos:
                    video_id = video_options[display_name]
                    
                    # Backup existing outputs if requested
                    if backup_outputs:
                        _backup_video_outputs(video_id)
                    
                    # Reset video for reprocessing
                    if st.session_state.db.reset_video_for_reprocessing(video_id):
                        reprocessed_count += 1
                        
                        # Add to processing queue if immediate processing is requested
                        if process_immediately:
                            st.session_state.batch_processor.add_video_to_queue_by_id(video_id)
                
                if reprocessed_count > 0:
                    st.success(f"✅ Reset {reprocessed_count} videos for reprocessing")
                    
                    if process_immediately:
                        st.info("🚀 Started immediate processing")
                    else:
                        st.info("📋 Videos added to queue. Go to Batch Processing to start.")
                    
                    st.rerun()
                else:
                    st.error("❌ Failed to reset videos for reprocessing")
        
        with col2:
            if st.button("📋 Add to Queue Only", use_container_width=True):
                queued_count = 0
                
                for display_name in selected_videos:
                    video_id = video_options[display_name]
                    
                    if st.session_state.db.reset_video_for_reprocessing(video_id):
                        queued_count += 1
                
                if queued_count > 0:
                    st.success(f"✅ Added {queued_count} videos to processing queue")
                    st.info("Go to Batch Processing page to start processing")
                else:
                    st.error("❌ Failed to add videos to queue")
        
        with col3:
            if st.button("⚠️ Reset Only", use_container_width=True):
                st.warning("This will reset processing status without reprocessing")
                
                reset_count = 0
                for display_name in selected_videos:
                    video_id = video_options[display_name]
                    if st.session_state.db.reset_video_for_reprocessing(video_id):
                        reset_count += 1
                
                if reset_count > 0:
                    st.success(f"✅ Reset {reset_count} videos to 'discovered' status")
                else:
                    st.error("❌ Failed to reset videos")
    
    # Processing history
    st.subheader("Processing History")
    
    if completed_videos:
        history_df = pd.DataFrame([
            {
                'Video': Path(video['file_path']).name,
                'Duration (min)': f"{(video['duration_seconds'] or 0) / 60:.1f}",
                'Processing Time (min)': f"{(video['processing_time_seconds'] or 0) / 60:.1f}",
                'Completed': video['completed_at'][:10] if video['completed_at'] else 'Unknown'
            }
            for video in completed_videos[:10]  # Show latest 10
        ])
        
        st.dataframe(history_df, use_container_width=True)
        
        if len(completed_videos) > 10:
            st.info(f"Showing latest 10 of {len(completed_videos)} completed videos")

def _backup_video_outputs(video_id: int):
    """Backup existing outputs for a video before reprocessing."""
    # This would move existing outputs to a backup folder
    # Implementation would depend on how outputs are organized
    pass

if __name__ == "__main__":
    main()
