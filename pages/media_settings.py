"""
Media settings page for configuring audio track and subtitle selection.
"""
import streamlit as st
from pathlib import Path
from core.media_selector import MediaTrackSelector

def show_batch_media_settings(video_list):
    """Show batch media settings for multiple videos."""
    
    st.subheader("🎵 Audio & Subtitle Settings")
    
    # Global strategy selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Audio Track Strategy:**")
        audio_strategy = st.selectbox(
            "How to select audio tracks:",
            [
                "Auto (Recommended)",
                "Prefer English",
                "Most Channels First", 
                "Manual Selection Per Video"
            ],
            help="Auto selects the best track automatically"
        )
    
    with col2:
        st.write("**Subtitle Strategy:**")
        subtitle_strategy = st.selectbox(
            "How to handle subtitles:",
            [
                "Auto Detect",
                "English Only", 
                "No Subtitles",
                "Manual Selection Per Video"
            ],
            help="Auto Detect finds the best available subtitles"
        )
    
    # Advanced options
    with st.expander("🔧 Advanced Audio/Subtitle Options"):
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Audio Preferences:**")
            preferred_languages = st.multiselect(
                "Preferred Languages",
                ["en", "es", "fr", "de", "it", "pt", "ja", "ko", "zh"],
                default=["en"],
                help="Language priority for audio tracks"
            )
            
            min_channels = st.slider(
                "Minimum Audio Channels", 
                1, 8, 2,
                help="Prefer tracks with at least this many channels"
            )
        
        with col2:
            st.write("**Subtitle Preferences:**")
            subtitle_formats = st.multiselect(
                "Allowed Subtitle Formats",
                ["srt", "vtt", "ass", "ssa"],
                default=["srt", "vtt"],
                help="Which subtitle formats to consider"
            )
            
            force_subtitles = st.checkbox(
                "Force Subtitles",
                help="Require subtitles for processing"
            )
    
    # Preview for first few videos
    if len(video_list) > 0:
        st.write("**Preview for first 3 videos:**")
        
        selector = MediaTrackSelector()
        
        for i, video in enumerate(video_list[:3]):
            with st.expander(f"📹 {Path(video['file_path']).name}"):
                try:
                    # Analyze tracks
                    tracks = selector.analyze_media_tracks(video['file_path'])
                    
                    if tracks.get('audio_tracks'):
                        st.write("**Available Audio Tracks:**")
                        for j, track in enumerate(tracks['audio_tracks']):
                            lang = track.get('language', 'unknown')
                            channels = track.get('channels', 'unknown')
                            codec = track.get('codec_name', 'unknown')
                            st.write(f"- Track {j}: {codec} ({channels} channels, {lang})")
                    
                    if tracks.get('subtitle_tracks'):
                        st.write("**Available Subtitles:**")
                        for j, sub in enumerate(tracks['subtitle_tracks']):
                            lang = sub.get('language', 'unknown')
                            format_name = sub.get('codec_name', 'unknown')
                            st.write(f"- Track {j}: {format_name} ({lang})")
                    
                    if tracks.get('external_subtitles'):
                        st.write("**External Subtitles:**")
                        for sub_file in tracks['external_subtitles']:
                            st.write(f"- {Path(sub_file).name}")
                
                except Exception as e:
                    st.warning(f"Could not analyze tracks: {e}")
    
    # Return settings dictionary
    return {
        'audio_strategy': audio_strategy,
        'subtitle_strategy': subtitle_strategy,
        'preferred_languages': preferred_languages,
        'min_channels': min_channels,
        'subtitle_formats': subtitle_formats,
        'force_subtitles': force_subtitles
    }

def show_media_settings_for_video(video_path):
    """Show individual media settings for a single video."""
    
    st.write(f"**Media Settings for {Path(video_path).name}:**")
    
    selector = MediaTrackSelector()
    
    try:
        # Analyze the video
        tracks = selector.analyze_media_tracks(video_path)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Audio Track Selection:**")
            
            if tracks.get('audio_tracks'):
                audio_options = []
                for i, track in enumerate(tracks['audio_tracks']):
                    lang = track.get('language', 'unknown')
                    channels = track.get('channels', 'unknown') 
                    codec = track.get('codec_name', 'unknown')
                    label = f"Track {i}: {codec} ({channels} ch, {lang})"
                    audio_options.append((i, label))
                
                # Auto-select best track
                recommended = selector.select_best_audio_track(tracks['audio_tracks'])
                recommended_idx = 0
                if recommended:
                    for i, track in enumerate(tracks['audio_tracks']):
                        if track == recommended:
                            recommended_idx = i
                            break
                
                selected_audio = st.selectbox(
                    "Choose Audio Track:",
                    options=[opt[0] for opt in audio_options],
                    format_func=lambda x: [opt[1] for opt in audio_options if opt[0] == x][0],
                    index=recommended_idx
                )
            else:
                st.warning("No audio tracks found")
                selected_audio = None
        
        with col2:
            st.write("**Subtitle Selection:**")
            
            subtitle_options = [(-1, "No Subtitles")]
            
            # Internal subtitles
            if tracks.get('subtitle_tracks'):
                for i, sub in enumerate(tracks['subtitle_tracks']):
                    lang = sub.get('language', 'unknown')
                    format_name = sub.get('codec_name', 'unknown')
                    label = f"Internal {i}: {format_name} ({lang})"
                    subtitle_options.append((f"internal_{i}", label))
            
            # External subtitles
            if tracks.get('external_subtitles'):
                for i, sub_file in enumerate(tracks['external_subtitles']):
                    label = f"External: {Path(sub_file).name}"
                    subtitle_options.append((f"external_{sub_file}", label))
            
            selected_subtitle = st.selectbox(
                "Choose Subtitles:",
                options=[opt[0] for opt in subtitle_options],
                format_func=lambda x: [opt[1] for opt in subtitle_options if opt[0] == x][0],
                index=0
            )
        
        # Show selected configuration
        st.info(f"Selected: Audio track {selected_audio}, Subtitles: {selected_subtitle}")
        
        return {
            'audio_track': selected_audio,
            'subtitle_track': selected_subtitle,
            'video_path': video_path
        }
        
    except Exception as e:
        st.error(f"Error analyzing video tracks: {e}")
        return {
            'audio_track': 0,
            'subtitle_track': -1,
            'video_path': video_path
        }

def show_media_settings_page():
    """Main media settings configuration page."""
    
    st.header("🎵 Media Track Settings")
    
    st.write("""
    Configure how the system selects audio tracks and subtitles from your video files.
    These settings apply to video processing and summarization.
    """)
    
    # Default strategies
    st.subheader("🎯 Default Selection Strategies")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Audio Track Strategy:**")
        default_audio = st.selectbox(
            "Default audio selection:",
            [
                "Auto (Best Quality)",
                "Prefer English",
                "Most Channels First",
                "First Track"
            ]
        )
        
        st.write("**Audio Quality Preferences:**")
        prefer_lossless = st.checkbox("Prefer lossless audio (FLAC, PCM)")
        min_sample_rate = st.slider("Minimum sample rate (Hz)", 16000, 96000, 44100)
    
    with col2:
        st.write("**Subtitle Strategy:**")
        default_subtitle = st.selectbox(
            "Default subtitle selection:",
            [
                "Auto Detect Best",
                "English Only",
                "External Files First", 
                "No Subtitles"
            ]
        )
        
        st.write("**Subtitle Format Preferences:**")
        subtitle_priority = st.multiselect(
            "Format priority (first = highest priority):",
            ["srt", "vtt", "ass", "ssa"],
            default=["srt", "vtt"]
        )
    
    # Language preferences
    st.subheader("🌍 Language Preferences")
    
    col1, col2 = st.columns(2)
    
    with col1:
        audio_languages = st.multiselect(
            "Preferred audio languages (priority order):",
            ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
            default=["en"]
        )
    
    with col2:
        subtitle_languages = st.multiselect(
            "Preferred subtitle languages:",
            ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
            default=["en"]
        )
    
    # Advanced settings
    st.subheader("🔧 Advanced Settings")
    
    with st.expander("🎛️ Advanced Audio/Video Options"):
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Audio Processing:**")
            normalize_audio = st.checkbox("Normalize audio levels", value=True)
            audio_bitrate_min = st.slider("Minimum audio bitrate (kbps)", 64, 320, 128)
            
            st.write("**Channel Configuration:**")
            downmix_surround = st.checkbox("Downmix surround to stereo", value=True)
            preserve_surround = st.checkbox("Preserve original surround when possible")
        
        with col2:
            st.write("**Subtitle Processing:**")
            extract_subtitles = st.checkbox("Extract subtitles for analysis", value=True)
            clean_subtitle_text = st.checkbox("Clean subtitle formatting", value=True)
            
            st.write("**External Files:**")
            search_subtitle_files = st.checkbox("Search for external subtitle files", value=True)
            subtitle_search_radius = st.slider("Search radius (parent directories)", 0, 3, 1)
    
    # Test section
    st.subheader("🧪 Test Settings")
    
    test_video = st.file_uploader(
        "Upload a test video to preview track selection:",
        type=['mp4', 'mkv', 'avi', 'mov'],
        help="Upload a video file to test your settings"
    )
    
    if test_video:
        # Save temporarily and analyze
        temp_path = Path("temp") / test_video.name
        temp_path.parent.mkdir(exist_ok=True)
        
        with open(temp_path, 'wb') as f:
            f.write(test_video.read())
        
        st.write("**Analysis Results:**")
        
        selector = MediaTrackSelector()
        try:
            tracks = selector.analyze_media_tracks(str(temp_path))
            
            # Show what would be selected
            if tracks.get('audio_tracks'):
                best_audio = selector.select_best_audio_track(tracks['audio_tracks'])
                st.success(f"Would select audio: {best_audio.get('codec_name', 'unknown')} ({best_audio.get('channels', 'unknown')} channels)")
            
            if tracks.get('subtitle_tracks') or tracks.get('external_subtitles'):
                best_subtitle = selector.select_best_subtitle_track(
                    tracks.get('subtitle_tracks', []),
                    tracks.get('external_subtitles', [])
                )
                if best_subtitle:
                    st.success(f"Would select subtitle: {best_subtitle}")
                else:
                    st.info("No suitable subtitles found")
            
        except Exception as e:
            st.error(f"Error analyzing test video: {e}")
        
        finally:
            # Clean up temp file
            try:
                temp_path.unlink()
            except:
                pass
    
    # Save settings
    if st.button("💾 Save Settings", type="primary"):
        settings = {
            'default_audio_strategy': default_audio,
            'default_subtitle_strategy': default_subtitle,
            'prefer_lossless': prefer_lossless,
            'min_sample_rate': min_sample_rate,
            'subtitle_priority': subtitle_priority,
            'audio_languages': audio_languages,
            'subtitle_languages': subtitle_languages,
            'normalize_audio': normalize_audio,
            'audio_bitrate_min': audio_bitrate_min,
            'downmix_surround': downmix_surround,
            'preserve_surround': preserve_surround,
            'extract_subtitles': extract_subtitles,
            'clean_subtitle_text': clean_subtitle_text,
            'search_subtitle_files': search_subtitle_files,
            'subtitle_search_radius': subtitle_search_radius
        }
        
        # Store in session state
        st.session_state.media_settings = settings
        st.success("Media settings saved!")