"""
Video Discovery page - moved from main app.py for proper Streamlit navigation.
"""
import streamlit as st
from pathlib import Path
import pandas as pd

def show_video_discovery():
    """Display video discovery and scanning interface."""
    st.header("🔍 Video Discovery")
    
    # Recent Folders section
    st.subheader("📂 Recent Folders")
    recent_folders = st.session_state.db.get_recent_folders(limit=5)
    
    if recent_folders:
        # Create columns for recent folder buttons
        cols = st.columns(min(len(recent_folders), 3))
        
        for i, folder in enumerate(recent_folders[:3]):  # Show max 3 in one row
            with cols[i]:
                folder_name = Path(folder['folder_path']).name or "Root"
                last_scan = folder['last_scanned'][:10] if folder['last_scanned'] else 'Unknown'
                button_text = f"📁 {folder_name}\n{folder['video_count']} videos • {last_scan}"
                
                if st.button(button_text, use_container_width=True, key=f"recent_{i}"):
                    st.session_state.scan_directory = folder['folder_path']
                    st.rerun()
        
        # Show more recent folders in expandable section
        if len(recent_folders) > 3:
            with st.expander(f"Show {len(recent_folders) - 3} more recent folders"):
                for i, folder in enumerate(recent_folders[3:], 3):
                    folder_name = Path(folder['folder_path']).name or "Root"
                    last_scan = folder['last_scanned'][:10] if folder['last_scanned'] else 'Unknown'
                    
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.write(f"📁 {folder_name}")
                    with col2:
                        st.write(f"{folder['video_count']} videos")
                    with col3:
                        if st.button("📁 Select", key=f"recent_select_{i}"):
                            st.session_state.scan_directory = folder['folder_path']
                            st.rerun()
    else:
        st.info("No recent folders found. Start by scanning a directory below.")
    
    st.divider()
    
    # Directory scanning section
    st.subheader("📁 Directory Scanning")
    
    # Text input for directory path
    scan_directory = st.text_input(
        "Enter directory path to scan:",
        value=st.session_state.get('scan_directory', ''),
        help="Enter the full path to the directory containing your video files"
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🔍 Scan Directory", use_container_width=True, disabled=not scan_directory):
            if scan_directory and Path(scan_directory).exists():
                st.session_state.scan_directory = scan_directory
                
                # Initialize batch processor
                from core.batch_processor import BatchProcessor
                batch_processor = BatchProcessor()
                
                with st.spinner("Scanning directory for videos..."):
                    try:
                        from core.video_discovery import VideoDiscovery
                        discovery = VideoDiscovery()
                        video_files_generator = discovery.scan_directory(scan_directory)
                        
                        # Convert generator to list to get count and store results
                        video_files = list(video_files_generator)
                        
                        if video_files:
                            # Add videos to database and get IDs
                            videos_with_ids = []
                            for video_metadata in video_files:
                                try:
                                    # Add video to database or get existing
                                    existing_video = st.session_state.db.get_video_by_path(video_metadata['file_path'])
                                    if existing_video:
                                        # Use existing video with database ID
                                        video_metadata.update(existing_video)
                                    else:
                                        # Add new video to database
                                        video_id = st.session_state.db.add_video(video_metadata['file_path'], video_metadata)
                                        video_metadata['id'] = video_id
                                        video_metadata['status'] = 'discovered'
                                    
                                    videos_with_ids.append(video_metadata)
                                except Exception as e:
                                    st.error(f"Error adding {video_metadata['file_path']} to database: {e}")
                            
                            st.success(f"Found {len(videos_with_ids)} video files!")
                            st.session_state.discovered_videos = videos_with_ids
                            st.session_state.scan_completed = True
                            st.session_state.videos_loaded = True  # Mark as loaded so it persists
                        else:
                            st.warning("No supported video files found in the specified directory.")
                            
                    except Exception as e:
                        st.error(f"Error scanning directory: {str(e)}")
                        
            else:
                st.error("Please enter a valid directory path")
    
    with col2:
        if st.button("📂 Browse Folders", use_container_width=True):
            # This would ideally open a file browser, but we'll provide common paths
            st.info("Common video directories:\n- /Movies\n- /Users/[username]/Movies\n- D:\\Movies")
    
    # Debug: Show current session state
    if st.checkbox("🔍 Debug Mode", help="Show debug information"):
        st.write("**Session State Debug:**")
        st.write(f"- videos_loaded: {st.session_state.get('videos_loaded', False)}")
        st.write(f"- scan_completed: {st.session_state.get('scan_completed', False)}")
        st.write(f"- discovered_videos count: {len(st.session_state.get('discovered_videos', []))}")
        st.write(f"- has db: {hasattr(st.session_state, 'db')}")
    
    # Auto-load videos from database on page load
    if not st.session_state.get('videos_loaded', False) and hasattr(st.session_state, 'db'):
        try:
            # Get all videos from database (get_videos_by_status() with no status returns all)
            all_videos = st.session_state.db.get_videos_by_status()
            st.write(f"**Database query returned {len(all_videos)} videos**")  # Debug output
            
            if all_videos:
                st.session_state.discovered_videos = all_videos
                st.session_state.scan_completed = True
                st.session_state.videos_loaded = True
                st.success(f"📚 Auto-loaded {len(all_videos)} videos from database")
            else:
                st.info("No videos found in database. Scan a directory to discover videos.")
        except Exception as e:
            st.error(f"Error loading videos from database: {e}")
    
    # Display discovered videos if available (from scan or database)
    videos = st.session_state.get('discovered_videos', [])
    st.write(f"**Videos to display: {len(videos)}**")  # Debug output
    
    if videos and len(videos) > 0:
        st.divider()
        st.subheader("🎬 Video Selection Interface")
        st.write(f"Select from {len(videos)} available videos:")
        
        # Create DataFrame for display
        video_data = []
        for video in videos:
            # Convert file_size from bytes to MB for display
            file_size_mb = video.get('file_size', 0) / (1024 * 1024)
            duration_min = video.get('duration_seconds', 0) / 60 if video.get('duration_seconds') else 0
            
            video_data.append({
                'Filename': Path(video['file_path']).name,
                'Size': f"{file_size_mb:.1f} MB",
                'Duration': f"{duration_min:.1f} min",
                'Status': video.get('status', 'Discovered').title(),
                'Path': video['file_path']
            })
        
        df = pd.DataFrame(video_data)
        
        # Initialize selection state (ensure it's always a set)
        if 'selected_videos' not in st.session_state or not isinstance(st.session_state.selected_videos, set):
            st.session_state.selected_videos = set()
        
        # Select All / None controls
        col1, col2, col3 = st.columns([2, 2, 6])
        with col1:
            if st.button("✅ Select All", use_container_width=True):
                # Ensure it's a set and add all video paths
                if not isinstance(st.session_state.selected_videos, set):
                    st.session_state.selected_videos = set()
                st.session_state.selected_videos.update(v['file_path'] for v in videos)
                st.rerun()
        with col2:
            if st.button("❌ Clear All", use_container_width=True):
                # Ensure it's a set before clearing
                if not isinstance(st.session_state.selected_videos, set):
                    st.session_state.selected_videos = set()
                else:
                    st.session_state.selected_videos.clear()
                st.rerun()
        with col3:
            selected_count = len(st.session_state.selected_videos)
            st.write(f"**{selected_count} videos selected**")
        
        st.divider()
        
        # Individual video selection with cleaner layout
        st.write("**Select videos to process:**")
        
        for i, video in enumerate(videos):
            # Create a container for each video
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([1, 4, 1.5, 1.5, 1.5])
                
                with col1:
                    # Checkbox for selection
                    video_path = video['file_path']
                    is_selected = video_path in st.session_state.selected_videos
                    
                    # Use a callback to handle checkbox changes properly
                    checkbox_key = f"select_{i}"
                    checkbox_value = st.checkbox("", value=is_selected, key=checkbox_key, label_visibility="collapsed")
                    
                    # Update selection based on checkbox state
                    if checkbox_value and video_path not in st.session_state.selected_videos:
                        st.session_state.selected_videos.add(video_path)
                    elif not checkbox_value and video_path in st.session_state.selected_videos:
                        st.session_state.selected_videos.remove(video_path)
                
                with col2:
                    st.write(f"**{Path(video['file_path']).name}**")
                
                with col3:
                    file_size_mb = video.get('file_size', 0) / (1024 * 1024)
                    st.write(f"{file_size_mb:.1f} MB")
                
                with col4:
                    duration_seconds = video.get('duration_seconds', 0)
                    if duration_seconds:
                        duration_str = f"{int(duration_seconds//60)}:{int(duration_seconds%60):02d}"
                    else:
                        duration_str = "Unknown"
                    st.write(duration_str)
                
                with col5:
                    status = video.get('status', 'discovered')
                    status_color = {
                        'discovered': '🔵',
                        'queued': '🟡', 
                        'processing': '🟠',
                        'completed': '🟢',
                        'error': '🔴'
                    }.get(status, '⚪')
                    st.write(f"{status_color} {status.title()}")
        
        # Action buttons
        st.divider()
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            selected_count = len(st.session_state.selected_videos)
            if st.button(f"➕ Add Selected ({selected_count})", use_container_width=True, disabled=selected_count == 0):
                added_count = 0
                for video_path in st.session_state.selected_videos:
                    try:
                        # Find the matching video to get proper ID
                        matching_video = next((v for v in videos if v['file_path'] == video_path), None)
                        if matching_video:
                            video_id = matching_video.get('id')
                            if video_id:
                                st.session_state.db.update_processing_status(video_id, 'queued')
                                added_count += 1
                            else:
                                st.error(f"No video ID found for {Path(video_path).name}")
                    except Exception as e:
                        st.error(f"Failed to add {Path(video_path).name}: {str(e)}")
                
                if added_count > 0:
                    st.success(f"Added {added_count} selected videos to processing queue!")
                    st.session_state.selected_videos.clear()  # Clear selection after adding
                    st.rerun()
        
        with col2:
            if st.button("➕ Add All to Queue", use_container_width=True):
                added_count = 0
                for video in videos:
                    try:
                        video_id = video.get('id')
                        if video_id:
                            st.session_state.db.update_processing_status(video_id, 'queued')
                            added_count += 1
                        else:
                            st.error(f"No video ID found for {Path(video['file_path']).name}")
                    except Exception as e:
                        st.error(f"Failed to add {Path(video['file_path']).name}: {str(e)}")
                
                if added_count > 0:
                    st.success(f"Added {added_count} videos to processing queue!")
                    st.rerun()
        
        with col3:
            if st.button("🔄 Refresh Status", use_container_width=True):
                # Clear auto-load flag to force refresh from database
                st.session_state.videos_loaded = False
                st.rerun()
        
        with col4:
            if st.button("🗑️ Clear Results", use_container_width=True):
                # Clear all session data related to video discovery
                for key in ['discovered_videos', 'scan_completed', 'videos_loaded']:
                    if key in st.session_state:
                        del st.session_state[key]
                # Ensure selected_videos is reset as an empty set
                st.session_state.selected_videos = set()
                st.rerun()
    
    # Instructions
    with st.expander("ℹ️ Instructions"):
        st.markdown("""
        **How to use Video Discovery:**
        
        1. **Recent Folders**: Click on any recently scanned folder to quickly access them
        2. **Directory Scanning**: Enter the full path to your video directory and click "Scan Directory"
        3. **Add to Queue**: Once videos are discovered, add them to the processing queue
        4. **Supported Formats**: MP4, MKV, AVI, MOV, WMV files
        
        **Common Issues:**
        - Ensure the directory path is correct and accessible
        - Check that video files are in supported formats
        - Large directories may take time to scan
        """)