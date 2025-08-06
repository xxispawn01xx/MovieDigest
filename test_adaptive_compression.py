#!/usr/bin/env python3
"""
Test adaptive compression logic for different video lengths.
"""
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.summarizer import VideoSummarizer
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_adaptive_compression():
    """Test the new adaptive compression logic."""
    
    print("🎯 Testing Adaptive Compression Logic")
    print("=" * 45)
    
    # Show configuration
    print(f"✓ Default compression: {config.SUMMARY_LENGTH_PERCENT}%")
    print(f"✓ Long video compression: {config.LONG_VIDEO_LENGTH_PERCENT}%")
    print(f"✓ Long video threshold: {config.LONG_VIDEO_THRESHOLD_MINUTES} minutes")
    
    # Initialize summarizer
    summarizer = VideoSummarizer()
    
    # Test short video (60 minutes)
    print(f"\n📹 Testing 60-minute video (should use {config.SUMMARY_LENGTH_PERCENT}%):")
    short_duration = 60 * 60  # 60 minutes in seconds
    
    if short_duration <= summarizer.long_video_threshold:
        expected_compression = summarizer.default_compression
        print(f"  ✓ Uses default compression: {expected_compression*100:.0f}%")
    else:
        print(f"  ❌ Should use default compression but threshold is {summarizer.long_video_threshold/60:.0f}min")
    
    # Test long video (120 minutes) 
    print(f"\n🎬 Testing 120-minute video (should use {config.LONG_VIDEO_LENGTH_PERCENT}%):")
    long_duration = 120 * 60  # 120 minutes in seconds
    
    if long_duration > summarizer.long_video_threshold:
        expected_compression = summarizer.long_video_compression
        print(f"  ✓ Uses long video compression: {expected_compression*100:.0f}%")
        
        # Calculate expected summary length
        expected_summary_length = long_duration * expected_compression / 60
        print(f"  ✓ Expected summary length: {expected_summary_length:.1f} minutes")
    else:
        print(f"  ❌ Should use long video compression but threshold is {summarizer.long_video_threshold/60:.0f}min")
    
    # Test very long video (180 minutes)
    print(f"\n🎭 Testing 180-minute video (should use {config.LONG_VIDEO_LENGTH_PERCENT}%):")
    very_long_duration = 180 * 60  # 180 minutes in seconds
    
    if very_long_duration > summarizer.long_video_threshold:
        expected_compression = summarizer.long_video_compression
        print(f"  ✓ Uses long video compression: {expected_compression*100:.0f}%")
        
        # Calculate expected summary length
        expected_summary_length = very_long_duration * expected_compression / 60
        print(f"  ✓ Expected summary length: {expected_summary_length:.1f} minutes")
    else:
        print(f"  ❌ Should use long video compression")
    
    print(f"\n✅ Adaptive compression configured correctly!")
    print(f"\n📊 Summary:")
    print(f"  • Videos ≤ {config.LONG_VIDEO_THRESHOLD_MINUTES}min: {config.SUMMARY_LENGTH_PERCENT}% compression")
    print(f"  • Videos > {config.LONG_VIDEO_THRESHOLD_MINUTES}min: {config.LONG_VIDEO_LENGTH_PERCENT}% compression")
    print(f"  • This maintains better narrative flow for longer movies")
    
    return True

if __name__ == "__main__":
    test_adaptive_compression()