"""
SQLite database management for video processing state tracking.
Maintains processing status, metadata, and validation scores.
"""
import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import config

logger = logging.getLogger(__name__)

class VideoDatabase:
    """Database manager for video processing state and metadata."""
    
    def __init__(self, db_path: Path = config.DATABASE_PATH):
        """Initialize database connection and create tables."""
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Create database tables if they don't exist."""
        if config.DISABLE_DATABASE_INIT:
            logger.info("Database initialization disabled")
            return
            
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS videos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT UNIQUE NOT NULL,
                    file_size INTEGER,
                    duration_seconds REAL,
                    resolution TEXT,
                    fps REAL,
                    has_subtitles BOOLEAN DEFAULT FALSE,
                    subtitle_path TEXT,
                    discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_modified TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS processing_status (
                    video_id INTEGER PRIMARY KEY,
                    status TEXT NOT NULL DEFAULT 'discovered',
                    progress_percent REAL DEFAULT 0.0,
                    current_stage TEXT,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    error_message TEXT,
                    processing_time_seconds REAL,
                    FOREIGN KEY (video_id) REFERENCES videos (id)
                );
                
                CREATE TABLE IF NOT EXISTS scene_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id INTEGER NOT NULL,
                    scene_number INTEGER NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL NOT NULL,
                    duration REAL NOT NULL,
                    importance_score REAL,
                    scene_type TEXT,
                    frame_path TEXT,
                    FOREIGN KEY (video_id) REFERENCES videos (id)
                );
                
                CREATE TABLE IF NOT EXISTS transcriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id INTEGER NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL NOT NULL,
                    text TEXT NOT NULL,
                    confidence REAL,
                    speaker_id TEXT,
                    FOREIGN KEY (video_id) REFERENCES videos (id)
                );
                
                CREATE TABLE IF NOT EXISTS summaries (
                    video_id INTEGER PRIMARY KEY,
                    summary_path TEXT NOT NULL,
                    summary_length_seconds REAL,
                    compression_ratio REAL,
                    key_scenes TEXT,  -- JSON array
                    narrative_analysis TEXT,
                    vlc_bookmark_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (video_id) REFERENCES videos (id)
                );
                
                CREATE TABLE IF NOT EXISTS validation_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id INTEGER NOT NULL,
                    metric_name TEXT NOT NULL,
                    score REAL NOT NULL,
                    benchmark_dataset TEXT,
                    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (video_id) REFERENCES videos (id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_videos_path ON videos(file_path);
                CREATE INDEX IF NOT EXISTS idx_processing_status ON processing_status(status);
                CREATE INDEX IF NOT EXISTS idx_scenes_video ON scene_data(video_id);
                CREATE INDEX IF NOT EXISTS idx_transcriptions_video ON transcriptions(video_id);
            """)
    
    def add_video(self, file_path: str, metadata: Dict) -> int:
        """Add a new video to the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO videos 
                (file_path, file_size, duration_seconds, resolution, fps, 
                 has_subtitles, subtitle_path, last_modified)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                file_path,
                metadata.get('file_size'),
                metadata.get('duration'),
                metadata.get('resolution'),
                metadata.get('fps'),
                metadata.get('has_subtitles', False),
                metadata.get('subtitle_path'),
                metadata.get('last_modified')
            ))
            
            video_id = cursor.lastrowid
            if video_id is None:
                raise ValueError("Failed to insert video")
            
            # Initialize processing status
            cursor.execute("""
                INSERT OR REPLACE INTO processing_status (video_id, status)
                VALUES (?, 'discovered')
            """, (video_id,))
            
            return video_id
    
    def get_videos_by_status(self, status: Optional[str] = None) -> List[Dict]:
        """Get videos filtered by processing status."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if status:
                cursor.execute("""
                    SELECT v.*, ps.status, ps.progress_percent, ps.current_stage
                    FROM videos v
                    JOIN processing_status ps ON v.id = ps.video_id
                    WHERE ps.status = ?
                    ORDER BY v.discovered_at DESC
                """, (status,))
            else:
                cursor.execute("""
                    SELECT v.*, ps.status, ps.progress_percent, ps.current_stage
                    FROM videos v
                    JOIN processing_status ps ON v.id = ps.video_id
                    ORDER BY v.discovered_at DESC
                """)
            
            return [dict(row) for row in cursor.fetchall()]
    
    def update_processing_status(self, video_id: int, status: str, 
                               progress: Optional[float] = None, stage: Optional[str] = None,
                               error_message: Optional[str] = None):
        """Update processing status for a video."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            updates = ["status = ?"]
            params: List = [status]
            
            if progress is not None:
                updates.append("progress_percent = ?")
                params.append(progress)
            
            if stage is not None:
                updates.append("current_stage = ?")
                params.append(stage)
            
            if error_message is not None:
                updates.append("error_message = ?")
                params.append(error_message)
            
            if status == 'processing' and progress == 0:
                updates.append("started_at = CURRENT_TIMESTAMP")
            elif status == 'completed':
                updates.append("completed_at = CURRENT_TIMESTAMP")
            
            params.append(video_id)
            
            cursor.execute(f"""
                UPDATE processing_status 
                SET {', '.join(updates)}
                WHERE video_id = ?
            """, params)
    
    def update_video_status(self, video_id: int, status: str):
        """Update the status of a video (alias for update_processing_status)."""
        self.update_processing_status(video_id, status)
    
    def add_scene_data(self, video_id: int, scenes: List[Dict]):
        """Add scene detection data for a video."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Clear existing scene data
            cursor.execute("DELETE FROM scene_data WHERE video_id = ?", (video_id,))
            
            # Insert new scene data
            for i, scene in enumerate(scenes):
                cursor.execute("""
                    INSERT INTO scene_data 
                    (video_id, scene_number, start_time, end_time, duration,
                     importance_score, scene_type, frame_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    video_id, i + 1,
                    scene['start_time'], scene['end_time'], scene['duration'],
                    scene.get('importance_score'),
                    scene.get('scene_type'),
                    scene.get('frame_path')
                ))
    
    def add_transcription_data(self, video_id: int, transcriptions: List[Dict]):
        """Add transcription data for a video."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Clear existing transcription data
            cursor.execute("DELETE FROM transcriptions WHERE video_id = ?", (video_id,))
            
            # Insert new transcription data
            for trans in transcriptions:
                cursor.execute("""
                    INSERT INTO transcriptions 
                    (video_id, start_time, end_time, text, confidence, speaker_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    video_id,
                    trans['start_time'], trans['end_time'], trans['text'],
                    trans.get('confidence'), trans.get('speaker_id')
                ))
    
    def add_summary_data(self, video_id: int, summary_data: Dict):
        """Add summary data for a video."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO summaries
                (video_id, summary_path, summary_length_seconds, compression_ratio,
                 key_scenes, narrative_analysis, vlc_bookmark_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                video_id,
                summary_data['summary_path'],
                summary_data['summary_length_seconds'],
                summary_data['compression_ratio'],
                json.dumps(summary_data.get('key_scenes', [])),
                summary_data.get('narrative_analysis'),
                summary_data.get('vlc_bookmark_path')
            ))
    
    def add_validation_score(self, video_id: int, metric_name: str, 
                           score: float, benchmark_dataset: Optional[str] = None):
        """Add validation score for a video."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO validation_scores
                (video_id, metric_name, score, benchmark_dataset)
                VALUES (?, ?, ?, ?)
            """, (video_id, metric_name, score, benchmark_dataset))
    
    def get_processing_stats(self) -> Dict:
        """Get overall processing statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    ps.status,
                    COUNT(*) as count,
                    AVG(ps.progress_percent) as avg_progress
                FROM processing_status ps
                GROUP BY ps.status
            """)
            
            stats = {}
            for row in cursor.fetchall():
                stats[row[0]] = {
                    'count': row[1],
                    'avg_progress': row[2] or 0.0
                }
            
            return stats
    
    def get_video_details(self, video_id: int) -> Optional[Dict]:
        """Get complete details for a specific video."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT v.*, ps.*, s.summary_path, s.vlc_bookmark_path
                FROM videos v
                JOIN processing_status ps ON v.id = ps.video_id
                LEFT JOIN summaries s ON v.id = s.video_id
                WHERE v.id = ?
            """, (video_id,))
            
            result = cursor.fetchone()
            return dict(result) if result else None
