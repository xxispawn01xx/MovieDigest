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
                            st.success(f"Found {len(video_files)} video files!")
                            st.session_state.discovered_videos = video_files
                            st.session_state.scan_completed = True
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
    
    # Display discovered videos if available
    if st.session_state.get('scan_completed', False) and st.session_state.get('discovered_videos'):
        st.divider()
        st.subheader("🎬 Discovered Videos")
        
        videos = st.session_state.discovered_videos
        
        # Create DataFrame for display
        video_data = []
        for video in videos:
            video_data.append({
                'Filename': Path(video['file_path']).name,
                'Size': f"{video['file_size_mb']:.1f} MB",
                'Duration': f"{video.get('duration_minutes', 0):.1f} min",
                'Status': video.get('status', 'Discovered').title(),
                'Path': video['file_path']
            })
        
        df = pd.DataFrame(video_data)
        
        # Display with selection
        st.dataframe(
            df[['Filename', 'Size', 'Duration', 'Status']], 
            use_container_width=True,
            hide_index=True
        )
        
        # Action buttons
        st.divider()
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("➕ Add All to Queue", use_container_width=True):
                from core.batch_processor import BatchProcessor
                batch_processor = BatchProcessor()
                
                added_count = 0
                for video in videos:
                    try:
                        success = st.session_state.db.update_video_status(video['file_path'], 'queued')
                        if success:
                            added_count += 1
                    except Exception as e:
                        st.error(f"Failed to add {Path(video['file_path']).name}: {str(e)}")
                
                if added_count > 0:
                    st.success(f"Added {added_count} videos to processing queue!")
                    st.rerun()
        
        with col2:
            if st.button("🔄 Refresh Status", use_container_width=True):
                st.rerun()
        
        with col3:
            if st.button("🗑️ Clear Results", use_container_width=True):
                if 'discovered_videos' in st.session_state:
                    del st.session_state.discovered_videos
                if 'scan_completed' in st.session_state:
                    del st.session_state.scan_completed
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