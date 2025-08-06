"""
Media settings page for audio/subtitle track selection.
"""
import streamlit as st
from pathlib import Path
from core.media_selector import MediaTrackSelector

def show_media_settings_for_video(video_path: str):
    """Show media track selection interface for a specific video."""
    
    st.subheader("🎵 Audio & Subtitle Selection")
    
    # Initialize media selector
    selector = MediaTrackSelector()
    
    # Get stream information
    ui_info = selector.get_stream_info_for_ui(video_path)
    
    if not ui_info.get('has_multiple_options', False):
        st.info("This video has standard single audio track configuration.")
        return {'audio_track': 0, 'subtitle_track': None}
    
    st.markdown(f"**Video:** `{Path(video_path).name}`")
    
    # Audio track selection
    st.markdown("### 🔊 Audio Track Selection")
    
    if len(ui_info['audio_options']) > 1:
        audio_descriptions = [opt['description'] for opt in ui_info['audio_options']]
        audio_values = [opt['value'] for opt in ui_info['audio_options']]
        
        default_audio_idx = 0
        recommended_audio = ui_info.get('recommended_audio', 0)
        
        # Find index of recommended audio
        for i, opt in enumerate(ui_info['audio_options']):
            if opt['value'] == recommended_audio:
                default_audio_idx = i
                break
        
        selected_audio_idx = st.selectbox(
            "Choose audio track:",
            range(len(audio_descriptions)),
            index=default_audio_idx,
            format_func=lambda x: audio_descriptions[x],
            help="Select which audio track to use for transcription and final summary"
        )
        
        selected_audio = audio_values[selected_audio_idx]
        
        if selected_audio == recommended_audio:
            st.success("✅ Using recommended audio track")
        else:
            st.info("ℹ️ Using custom audio track selection")
            
    else:
        selected_audio = ui_info['audio_options'][0]['value'] if ui_info['audio_options'] else 0
        st.write("Single audio track detected - using default")
    
    # Subtitle track selection
    st.markdown("### 📝 Subtitle Selection")
    
    if len(ui_info['subtitle_options']) > 1:
        subtitle_descriptions = [opt['description'] for opt in ui_info['subtitle_options']]
        subtitle_values = [opt['value'] for opt in ui_info['subtitle_options']]
        
        default_subtitle_idx = 0
        recommended_subtitle = ui_info.get('recommended_subtitle')
        
        # Find index of recommended subtitle
        if recommended_subtitle:
            for i, opt in enumerate(ui_info['subtitle_options']):
                if opt['value'] == recommended_subtitle:
                    default_subtitle_idx = i
                    break
        
        selected_subtitle_idx = st.selectbox(
            "Choose subtitle track:",
            range(len(subtitle_descriptions)),
            index=default_subtitle_idx,
            format_func=lambda x: subtitle_descriptions[x],
            help="Select subtitle track to include in summary (optional)"
        )
        
        selected_subtitle = subtitle_values[selected_subtitle_idx]
        
        if selected_subtitle == recommended_subtitle:
            st.success("✅ Using recommended subtitle track")
        elif selected_subtitle is None:
            st.info("ℹ️ No subtitles will be included")
        else:
            st.info("ℹ️ Using custom subtitle selection")
            
    else:
        selected_subtitle = None
        st.write("No subtitles available")
    
    # Preview section
    with st.expander("🔍 Track Information Preview"):
        st.markdown("**Selected Configuration:**")
        st.write(f"- Audio Track: {audio_descriptions[selected_audio_idx] if 'selected_audio_idx' in locals() else 'Default'}")
        st.write(f"- Subtitle Track: {subtitle_descriptions[selected_subtitle_idx] if 'selected_subtitle_idx' in locals() and selected_subtitle else 'None'}")
    
    return {
        'audio_track': selected_audio,
        'subtitle_track': selected_subtitle
    }

def show_batch_media_settings(video_list):
    """Show media settings for batch processing."""
    
    st.subheader("🎵 Batch Audio & Subtitle Settings")
    
    st.markdown("### Default Track Selection Strategy")
    
    col1, col2 = st.columns(2)
    
    with col1:
        audio_strategy = st.selectbox(
            "Audio Track Strategy:",
            [
                "Recommended (Default/English/Most Channels)",
                "Always First Track",
                "Prefer English",
                "Most Channels (Surround Sound)",
                "Manual Selection Per Video"
            ],
            help="How to automatically select audio tracks for batch processing"
        )
    
    with col2:
        subtitle_strategy = st.selectbox(
            "Subtitle Strategy:",
            [
                "Recommended (Default/English)",
                "No Subtitles", 
                "Always First Available",
                "Prefer English",
                "Manual Selection Per Video"
            ],
            help="How to handle subtitles in batch processing"
        )
    
    # Preview settings for first few videos
    if video_list and len(video_list) > 0:
        st.markdown("### 📋 Preview: First 3 Videos")
        
        selector = MediaTrackSelector()
        
        for i, video in enumerate(video_list[:3]):
            with st.expander(f"📹 {Path(video['file_path']).name}"):
                ui_info = selector.get_stream_info_for_ui(video['file_path'])
                
                if ui_info.get('has_multiple_options'):
                    st.write("**Available Audio Tracks:**")
                    for audio_opt in ui_info['audio_options']:
                        marker = "🎯" if audio_opt['value'] == ui_info.get('recommended_audio') else "  "
                        st.write(f"{marker} {audio_opt['description']}")
                    
                    if len(ui_info['subtitle_options']) > 1:
                        st.write("**Available Subtitle Tracks:**")
                        for sub_opt in ui_info['subtitle_options']:
                            marker = "🎯" if sub_opt['value'] == ui_info.get('recommended_subtitle') else "  "
                            st.write(f"{marker} {sub_opt['description']}")
                else:
                    st.info("Standard single-track configuration")
    
    return {
        'audio_strategy': audio_strategy,
        'subtitle_strategy': subtitle_strategy
    }