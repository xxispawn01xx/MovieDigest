#!/usr/bin/env python3
"""
Demo data generator for testing the Video Summarization Engine.
Creates sample video entries in the database to demonstrate export functionality.
"""
import sys
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from core.database import VideoDatabase

def create_demo_video_data():
    """Create demo video data for testing export functionality."""
    db = VideoDatabase()
    
    # Sample video metadata
    demo_video = {
        'file_path': '/demo/sample_movie.mp4',
        'title': 'Demo Action Movie',
        'duration': 7200,  # 2 hours
        'total_scenes': 45,
        'status': 'completed',
        'processing_time_seconds': 1800,  # 30 minutes
        'validation_f1': 0.857,
        'scenes': [
            {'start_time': 0, 'end_time': 120, 'scene_type': 'establishing'},
            {'start_time': 120, 'end_time': 300, 'scene_type': 'dialogue'},
            {'start_time': 300, 'end_time': 480, 'scene_type': 'action'},
            {'start_time': 480, 'end_time': 600, 'scene_type': 'transition'},
            {'start_time': 600, 'end_time': 900, 'scene_type': 'character_development'},
            # ... more scenes would be here
        ],
        'narrative_analysis': {
            'structure_analysis': {
                'act_structure': '5-act structure',
                'pacing': 'Fast-paced action thriller',
                'tone': 'Intense with comedic relief',
                'genre_indicators': ['action', 'thriller', 'adventure']
            },
            'key_moments': [
                {
                    'type': 'inciting_incident',
                    'description': 'Hero discovers the conspiracy',
                    'timestamp': 480,
                    'importance': 0.95
                },
                {
                    'type': 'plot_point_1',
                    'description': 'First major action sequence',
                    'timestamp': 1200,
                    'importance': 0.90
                },
                {
                    'type': 'midpoint',
                    'description': 'Hero faces betrayal',
                    'timestamp': 3600,
                    'importance': 0.92
                },
                {
                    'type': 'climax',
                    'description': 'Final confrontation',
                    'timestamp': 6000,
                    'importance': 1.0
                },
                {
                    'type': 'resolution',
                    'description': 'Hero saves the day',
                    'timestamp': 6800,
                    'importance': 0.85
                }
            ],
            'character_analysis': {
                'main_characters': ['Hero', 'Villain', 'Sidekick', 'Love Interest'],
                'character_arcs': ['Hero journey', 'Villain redemption'],
                'themes': ['Good vs Evil', 'Friendship', 'Sacrifice']
            },
            'narrative_summary': {
                'plot_summary': 'An action-packed thriller where a reluctant hero discovers a conspiracy and must overcome betrayal to save the world.',
                'structure_notes': 'Classic three-act structure with strong character development'
            }
        },
        'transcription': [
            {'start_time': 0, 'end_time': 5, 'text': 'In a world where nothing is as it seems...'},
            {'start_time': 480, 'end_time': 485, 'text': 'The files... they prove everything!'},
            {'start_time': 3600, 'end_time': 3605, 'text': 'You were working for them all along!'},
            {'start_time': 6000, 'end_time': 6005, 'text': 'This ends now!'}
        ],
        'summary': {
            'plot_summary': 'Demo movie showcasing the video summarization system capabilities',
            'key_scenes': [0, 8, 22, 40, 44],
            'compression_ratio': 0.15
        },
        'validation_metrics': {
            'f1_score': 0.857,
            'precision': 0.821,
            'recall': 0.896,
            'benchmark_comparison': 'Exceeds TVSum baseline'
        }
    }
    
    # Add to database
    try:
        video_id = db.add_video(demo_video['file_path'], demo_video)
        
        # Update with processed status and additional details
        db.update_video_status(video_id, 'completed', {
            'processing_time_seconds': demo_video['processing_time_seconds'],
            'validation_f1': demo_video['validation_f1'],
            'total_scenes': demo_video['total_scenes'],
            'narrative_analysis': json.dumps(demo_video['narrative_analysis']),
            'transcription': json.dumps(demo_video['transcription']),
            'summary': json.dumps(demo_video['summary']),
            'validation_metrics': json.dumps(demo_video['validation_metrics'])
        })
        
        print(f"✅ Demo video created with ID: {video_id}")
        print(f"📊 Video details: {demo_video['title']}")
        print(f"⏱️  Duration: {demo_video['duration']/60:.1f} minutes")
        print(f"🎬 Scenes: {demo_video['total_scenes']}")
        print(f"🎯 F1 Score: {demo_video['validation_f1']:.3f}")
        
        return video_id
        
    except Exception as e:
        print(f"❌ Failed to create demo data: {e}")
        return None

if __name__ == "__main__":
    create_demo_video_data()