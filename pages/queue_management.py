"""
Enhanced Queue Management page with individual video removal and progress tracking.
"""
import streamlit as st
st.set_page_config(page_title="Queue Management - Video Summarization Engine", layout="wide")
import pandas as pd
from pathlib import Path
import time
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def show_queue_management():
    """Display enhanced queue management with individual video controls and progress."""
    st.header("📋 Processing Queue Management")
    
    # Get queue status
    if hasattr(st.session_state, 'batch_processor'):
        queue_summary = st.session_state.batch_processor.get_queue_summary()
        is_processing = st.session_state.batch_processor.is_processing()
    else:
        queue_summary = {'total_in_queue': 0, 'processing': 0, 'completed': 0}
        is_processing = False
    
    # Queue overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Videos in Queue", queue_summary.get('total_in_queue', 0))
    
    with col2:
        st.metric("Currently Processing", queue_summary.get('processing', 0))
    
    with col3:
        st.metric("Completed Today", queue_summary.get('completed', 0))
    
    with col4:
        if is_processing:
            st.metric("Status", "🔄 Processing")
        else:
            st.metric("Status", "⏸️ Idle")
    
    # Queue controls
    st.subheader("Queue Controls")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("▶️ Start Processing", use_container_width=True, disabled=is_processing):
            if hasattr(st.session_state, 'batch_processor'):
                st.session_state.batch_processor.start_processing()
                st.success("Processing started!")
                st.rerun()
    
    with col2:
        if st.button("⏸️ Pause Processing", use_container_width=True, disabled=not is_processing):
            if hasattr(st.session_state, 'batch_processor'):
                st.session_state.batch_processor.pause_processing()
                st.success("Processing paused!")
                st.rerun()
    
    with col3:
        if st.button("🛑 Stop Processing", use_container_width=True, disabled=not is_processing):
            if hasattr(st.session_state, 'batch_processor'):
                st.session_state.batch_processor.stop_processing()
                st.success("Processing stopped!")
                st.rerun()
    
    # Live progress for currently processing videos
    processing_videos = st.session_state.db.get_videos_by_status(['processing'])
    if processing_videos:
        st.subheader("🔄 Live Processing Progress")
        
        # Create a container for real-time updates
        progress_container = st.container()
        
        with progress_container:
            for video in processing_videos:
                show_video_progress_card(video)
    
    # Queue management table
    st.subheader("Queue Management")
    
    # Get all videos with various statuses
    all_videos = st.session_state.db.get_videos_by_status(['discovered', 'queued', 'processing', 'completed', 'failed'])
    
    if all_videos:
        # Create enhanced dataframe with individual controls
        videos_data = []
        for video in all_videos:
            videos_data.append({
                'Select': False,
                'Filename': Path(video['file_path']).name,
                'Status': video['status'].title(),
                'Progress': video['progress_percent'],
                'Stage': video.get('current_stage', 'N/A'),
                'Size (MB)': f"{(video.get('file_size') or 0) / (1024*1024):.1f}",
                'Duration (min)': f"{(video.get('duration_seconds') or 0) / 60:.1f}",
                'Video ID': video['id'],
                'File Path': video['file_path']
            })
        
        videos_df = pd.DataFrame(videos_data)
        
        # Enhanced data editor with progress bars
        edited_df = st.data_editor(
            videos_df,
            column_config={
                'Select': st.column_config.CheckboxColumn('Select'),
                'Progress': st.column_config.ProgressColumn(
                    'Progress %',
                    help="Processing progress percentage",
                    min_value=0,
                    max_value=100,
                    format="%.1f%%"
                ),
                'File Path': None,  # Hide file path
                'Video ID': None   # Hide video ID
            },
            disabled=['Filename', 'Status', 'Progress', 'Stage', 'Size (MB)', 'Duration (min)'],
            use_container_width=True,
            key="queue_table"
        )
        
        # Update selected videos
        st.session_state.selected_videos = set(
            row['Video ID'] for _, row in edited_df.iterrows() if row['Select'] == True
        )
        
        # Bulk actions
        if st.session_state.selected_videos:
            st.info(f"Selected {len(st.session_state.selected_videos)} videos")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("📝 Add to Queue", use_container_width=True):
                    for video_id in st.session_state.selected_videos:
                        st.session_state.db.update_processing_status(video_id, 'queued')
                    st.success(f"Added {len(st.session_state.selected_videos)} videos to queue")
                    time.sleep(1)
                    st.rerun()
            
            with col2:
                if st.button("🔄 Reset for Reprocessing", use_container_width=True):
                    reset_count = 0
                    for video_id in st.session_state.selected_videos:
                        if st.session_state.db.reset_video_for_reprocessing(video_id):
                            reset_count += 1
                    st.success(f"Reset {reset_count} videos for reprocessing")
                    time.sleep(1)
                    st.rerun()
            
            with col3:
                if st.button("🗑️ Remove Selected", use_container_width=True):
                    with st.expander("⚠️ Confirm Removal", expanded=True):
                        st.warning("This will permanently remove the selected videos and all associated data from the database. The video files themselves will not be deleted.")
                        
                        col_confirm, col_cancel = st.columns(2)
                        with col_confirm:
                            if st.button("✅ Confirm Remove", type="primary"):
                                removed_count = 0
                                for video_id in st.session_state.selected_videos:
                                    if st.session_state.db.remove_video_completely(video_id):
                                        removed_count += 1
                                
                                if removed_count > 0:
                                    st.success(f"Successfully removed {removed_count} videos from database")
                                    st.session_state.selected_videos = set()
                                    time.sleep(2)
                                    st.rerun()
                                else:
                                    st.error("Failed to remove selected videos")
                        
                        with col_cancel:
                            if st.button("❌ Cancel"):
                                st.rerun()
            
            with col4:
                if st.button("ℹ️ View Details", use_container_width=True):
                    selected_video_id = next(iter(st.session_state.selected_videos))
                    show_video_details_modal(selected_video_id)
        
        # Filter controls
        with st.sidebar:
            st.subheader("🔍 Filter Videos")
            
            status_filter = st.multiselect(
                "Filter by Status",
                options=['Discovered', 'Queued', 'Processing', 'Completed', 'Failed'],
                default=['Discovered', 'Queued', 'Processing']
            )
            
            # Apply filters (in a real implementation, this would filter the query)
            if status_filter:
                st.info(f"Showing videos with status: {', '.join(status_filter)}")
    
    else:
        st.info("No videos found. Use the Video Discovery page to scan for videos.")
    
    # Auto-refresh for live updates
    if processing_videos:
        time.sleep(2)
        st.rerun()

def show_video_progress_card(video):
    """Display a progress card for an individual video."""
    filename = Path(video['file_path']).name
    progress = video['progress_percent']
    stage = video.get('current_stage', 'Unknown')
    
    with st.container():
        col1, col2 = st.columns([4, 1])
        
        with col1:
            st.write(f"**{filename}**")
            st.progress(progress / 100.0)
            st.caption(f"Current stage: {stage}")
        
        with col2:
            st.metric("Progress", f"{progress:.1f}%")
        
        # Show detailed stage progress if available
        if hasattr(st.session_state, 'progress_tracker'):
            stage_progress = st.session_state.progress_tracker.get_individual_video_progress(video['id'])
            if stage_progress and stage_progress.get('stages'):
                with st.expander("Stage Details", expanded=False):
                    for stage_name, stage_info in stage_progress['stages'].items():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"{stage_info['display_name']}")
                            st.progress(stage_info['progress_percent'] / 100.0)
                        with col2:
                            st.write(f"{stage_info['progress_percent']:.0f}%")
        
        st.divider()

def show_video_details_modal(video_id):
    """Show detailed information about a video."""
    video_details = st.session_state.db.get_video_details(video_id)
    
    if video_details:
        with st.expander("📊 Video Details", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**File Information**")
                st.write(f"Filename: {Path(video_details['file_path']).name}")
                st.write(f"Size: {(video_details.get('file_size', 0) / (1024*1024)):.1f} MB")
                st.write(f"Duration: {(video_details.get('duration_seconds', 0) / 60):.1f} minutes")
                st.write(f"Resolution: {video_details.get('resolution', 'Unknown')}")
            
            with col2:
                st.write("**Processing Information**")
                st.write(f"Status: {video_details.get('status', 'Unknown').title()}")
                st.write(f"Progress: {video_details.get('progress_percent', 0):.1f}%")
                st.write(f"Current Stage: {video_details.get('current_stage', 'N/A')}")
                
                if video_details.get('error_message'):
                    st.error(f"Error: {video_details['error_message']}")

# Initialize session state
if 'db' not in st.session_state:
    from core.database import Database
    st.session_state.db = Database()

if 'selected_videos' not in st.session_state:
    st.session_state.selected_videos = set()

# Initialize batch processor if not exists
if 'batch_processor' not in st.session_state:
    try:
        from core.batch_processor import BatchProcessor
        st.session_state.batch_processor = BatchProcessor(st.session_state.db)
    except Exception as e:
        st.error(f"Failed to initialize batch processor: {e}")

# Show the queue management interface
show_queue_management()