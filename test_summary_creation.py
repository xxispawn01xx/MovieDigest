#!/usr/bin/env python3
"""
Test script to demonstrate video summary creation and troubleshoot issues.
"""
import sys
from pathlib import Path
import logging

# Add project root to path
sys.path.append('.')

from core.summarizer import VideoSummarizer
from core.scene_detector import SceneDetector
from core.transcription import OfflineTranscriber
from core.narrative_analyzer import NarrativeAnalyzer
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_summary_creation():
    """Test the complete summary creation pipeline."""
    
    # Check if we have any videos in the database to test with
    from core.database import VideoDatabase
    db = VideoDatabase()
    
    # Get a processed video
    videos = db.get_videos_by_status()
    if not videos:
        print("❌ No videos found in database. Please scan some videos first.")
        return False
    
    # Find a video with some processing data
    test_video = None
    for video in videos:
        if Path(video['file_path']).exists():
            test_video = video
            break
    
    if not test_video:
        print("❌ No accessible video files found.")
        return False
    
    video_path = test_video['file_path']
    print(f"🎬 Testing with: {Path(video_path).name}")
    
    # Create test scene data (simplified for testing)
    test_scenes = [
        {
            'scene_number': 1,
            'start_time': 60.0,   # 1 minute
            'end_time': 120.0,    # 2 minutes
            'duration': 60.0,
            'importance_score': 0.8
        },
        {
            'scene_number': 2,
            'start_time': 300.0,  # 5 minutes
            'end_time': 360.0,    # 6 minutes
            'duration': 60.0,
            'importance_score': 0.9
        },
        {
            'scene_number': 3,
            'start_time': 600.0,  # 10 minutes
            'end_time': 660.0,    # 11 minutes
            'duration': 60.0,
            'importance_score': 0.7
        }
    ]
    
    # Create test transcription data
    test_transcription = [
        {
            'start_time': 60.0,
            'end_time': 120.0,
            'text': 'This is important dialogue in the first scene.',
            'confidence': 0.95
        },
        {
            'start_time': 300.0,
            'end_time': 360.0,
            'text': 'A crucial moment in the story happens here.',
            'confidence': 0.92
        }
    ]
    
    # Create test narrative analysis
    test_narrative = {
        'structure_analysis': {
            'act_breaks': [0.25, 0.75],
            'climax_timestamp': 600.0,
            'resolution_timestamp': 800.0
        },
        'key_moments': [
            {
                'timestamp': 120.0,
                'importance': 0.9,
                'description': 'Opening character introduction'
            },
            {
                'timestamp': 360.0,
                'importance': 0.95,
                'description': 'Major plot point'
            }
        ]
    }
    
    # Test the summarizer
    try:
        summarizer = VideoSummarizer()
        print("✅ VideoSummarizer initialized")
        
        # Test summary creation
        print("🔄 Creating video summary...")
        summary_result = summarizer.create_summary(
            video_path, test_scenes, test_transcription, test_narrative
        )
        
        if summary_result and summary_result.get('summary_path'):
            summary_path = Path(summary_result['summary_path'])
            if summary_path.exists():
                size_mb = summary_path.stat().st_size / (1024 * 1024)
                print(f"✅ Video summary created: {summary_path.name} ({size_mb:.1f} MB)")
                print(f"📁 Location: {summary_path}")
                return True
            else:
                print(f"❌ Summary file not found: {summary_path}")
        else:
            print("❌ Summary creation returned no result")
            
    except Exception as e:
        print(f"❌ Summary creation failed: {e}")
        logger.exception("Full error details:")
        
    return False

def test_ffmpeg_availability():
    """Test if FFmpeg is available and working."""
    import subprocess
    
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ FFmpeg is available")
            version_line = result.stdout.split('\n')[0]
            print(f"   {version_line}")
            return True
        else:
            print("❌ FFmpeg returned error")
            print(f"   {result.stderr}")
    except FileNotFoundError:
        print("❌ FFmpeg not found in PATH")
    except Exception as e:
        print(f"❌ FFmpeg test failed: {e}")
        
    return False

def main():
    """Main test function."""
    print("🧪 Video Summary Creation Test")
    print("=" * 40)
    
    # Test FFmpeg
    if not test_ffmpeg_availability():
        return False
    
    # Test output directories
    output_dir = Path("output/summaries")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Output directory ready: {output_dir}")
    
    # Test summary creation
    success = test_summary_creation()
    
    if success:
        print("\n🎉 Summary creation test PASSED!")
        print("📂 Check the output/summaries/ directory for your video summary")
    else:
        print("\n❌ Summary creation test FAILED!")
        print("🔧 Please check the logs above for specific errors")
    
    return success

if __name__ == "__main__":
    main()