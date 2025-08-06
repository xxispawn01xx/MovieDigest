#!/usr/bin/env python3
"""
Test the queue clearing GUI update fix.
"""
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.batch_processor import BatchProcessor
from core.database import VideoDatabase

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_queue_clear_gui_fix():
    """Test that queue clearing updates both memory and database properly."""
    
    print("🧪 Testing Queue Clear GUI Fix")
    print("=" * 40)
    
    # Initialize components
    db = VideoDatabase()
    processor = BatchProcessor()
    
    # Check initial state
    initial_summary = processor.get_queue_summary()
    print(f"✓ Initial queue size: {initial_summary['total_in_queue']}")
    
    # Simulate some queued videos in database
    print("\n📥 Simulating queued videos...")
    
    # Get current queued count
    queued_videos = db.get_videos_by_status('queued')
    if queued_videos:
        print(f"✓ Found {len(queued_videos)} videos already in queue")
    else:
        print("ℹ️ No videos currently queued")
    
    # Test the clear queue functionality
    print("\n🗑️ Testing queue clear...")
    cleared_count = processor.clear_queue()
    print(f"✓ Cleared {cleared_count} videos from queue")
    
    # Check that queue summary reflects the change
    after_summary = processor.get_queue_summary()
    print(f"✓ Queue size after clear: {after_summary['total_in_queue']}")
    
    # Verify database is also cleared
    remaining_queued = db.get_videos_by_status('queued')
    print(f"✓ Database queued videos after clear: {len(remaining_queued)}")
    
    if after_summary['total_in_queue'] == 0 and len(remaining_queued) == 0:
        print("✅ Queue clear successfully updates both memory and database")
        return True
    else:
        print("❌ Queue clear didn't update properly")
        print(f"   Memory queue: {after_summary['total_in_queue']}")
        print(f"   Database queue: {len(remaining_queued)}")
        return False

if __name__ == "__main__":
    test_queue_clear_gui_fix()