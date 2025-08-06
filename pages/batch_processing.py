"""
Enhanced batch processing page with audio/subtitle selection support.
"""
import streamlit as st
from pathlib import Path
from core.batch_processor import BatchProcessor
from core.media_selector import MediaTrackSelector
from pages.media_settings import show_batch_media_settings, show_media_settings_for_video

def show_batch_processing():
    """Display the batch processing interface with media track selection."""
    
    st.header("⚡ Batch Processing")
    
    # Initialize batch processor
    batch_processor = BatchProcessor()
    
    # Get current queue status
    queue_status = batch_processor.get_queue_status()
    current_queue = queue_status.get('queued_videos', [])
    
    # Display queue status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Queued Videos", len(current_queue))
    
    with col2:
        processing_count = len([v for v in current_queue if v.get('status') == 'processing'])
        st.metric("Processing", processing_count)
    
    with col3:
        completed_count = len([v for v in current_queue if v.get('status') == 'completed'])
        st.metric("Completed", completed_count)
    
    # Audio/Subtitle Settings
    if current_queue:
        st.divider()
        media_settings = show_batch_media_settings(current_queue)
        
        # Store settings in session state
        st.session_state.batch_media_settings = media_settings
    
    # Queue Management
    st.divider()
    st.subheader("📋 Processing Queue")
    
    if current_queue:
        # Display current queue
        for i, video in enumerate(current_queue):
            with st.expander(f"🎬 {Path(video['file_path']).name} - {video.get('status', 'queued').title()}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**Path:** `{video['file_path']}`")
                    st.write(f"**Status:** {video.get('status', 'queued').title()}")
                    if video.get('progress_percent'):
                        st.progress(video['progress_percent'] / 100.0)
                    
                    # Show individual media settings if strategy is manual
                    if (st.session_state.get('batch_media_settings', {}).get('audio_strategy') == 'Manual Selection Per Video' or
                        st.session_state.get('batch_media_settings', {}).get('subtitle_strategy') == 'Manual Selection Per Video'):
                        
                        with st.expander("🎵 Individual Media Settings"):
                            individual_settings = show_media_settings_for_video(video['file_path'])
                            # Store individual settings
                            if 'individual_media_settings' not in st.session_state:
                                st.session_state.individual_media_settings = {}
                            st.session_state.individual_media_settings[video['file_path']] = individual_settings
                
                with col2:
                    if st.button(f"🗑️ Remove", key=f"remove_{i}"):
                        try:
                            batch_processor.remove_from_queue(video['file_path'])
                            st.success("Removed from queue")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to remove: {e}")
        
        # Batch controls
        st.divider()
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("▶️ Start Processing", use_container_width=True):
                try:
                    # Apply media settings to batch processor
                    if hasattr(st.session_state, 'batch_media_settings'):
                        batch_processor.set_media_settings(st.session_state.batch_media_settings)
                    
                    if hasattr(st.session_state, 'individual_media_settings'):
                        batch_processor.set_individual_media_settings(st.session_state.individual_media_settings)
                    
                    batch_processor.start_processing()
                    st.success("Processing started!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to start processing: {e}")
        
        with col2:
            if st.button("⏸️ Pause Processing", use_container_width=True):
                try:
                    batch_processor.pause_processing()
                    st.info("Processing paused")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to pause: {e}")
        
        with col3:
            if st.button("🔄 Resume Processing", use_container_width=True):
                try:
                    batch_processor.resume_processing()
                    st.info("Processing resumed")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to resume: {e}")
        
        with col4:
            if st.button("🛑 Stop Processing", use_container_width=True):
                try:
                    batch_processor.stop_processing()
                    st.info("Processing stopped")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to stop: {e}")
        
        # Clear queue option
        st.divider()
        if st.button("🗑️ Clear Entire Queue", use_container_width=True):
            try:
                batch_processor.clear_queue()
                st.success("Queue cleared")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to clear queue: {e}")
                
    else:
        st.info("No videos in processing queue. Use Video Discovery to add videos.")
        
        # Quick add section
        st.subheader("➕ Quick Add Video")
        
        video_path = st.text_input(
            "Enter video file path:",
            help="Full path to a video file to add to the queue"
        )
        
        if st.button("Add to Queue") and video_path:
            if Path(video_path).exists():
                try:
                    if batch_processor.add_to_queue(video_path):
                        st.success(f"Added {Path(video_path).name} to queue!")
                        st.rerun()
                    else:
                        st.warning("Video already in queue")
                except Exception as e:
                    st.error(f"Failed to add video: {e}")
            else:
                st.error("File does not exist")
    
    # Processing Status
    if batch_processor.is_processing():
        st.divider()
        st.subheader("🔄 Current Processing Status")
        
        current_video = batch_processor.get_current_video()
        if current_video:
            st.write(f"**Currently Processing:** {Path(current_video['file_path']).name}")
            
            progress = current_video.get('progress_percent', 0)
            st.progress(progress / 100.0)
            st.write(f"Progress: {progress:.1f}%")
            
            current_stage = current_video.get('current_stage', 'Unknown')
            st.write(f"Stage: {current_stage}")
    
    # Auto-refresh option
    st.divider()
    auto_refresh = st.checkbox("🔄 Auto-refresh (5 seconds)", value=False)
    
    if auto_refresh:
        import time
        time.sleep(5)
        st.rerun()