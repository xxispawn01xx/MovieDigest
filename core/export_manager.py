"""
Export manager for generating different output formats from video summaries.
Supports VLC bookmarks, JSON exports, and summary reports.
"""
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ExportManager:
    """Manages export of video summaries to various formats."""
    
    def __init__(self, output_dir: str = "output"):
        """Initialize export manager."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def export_summary_json(self, video_data: Dict, output_path: Optional[str] = None) -> str:
        """
        Export complete video summary as JSON.
        
        Args:
            video_data: Complete video analysis data
            output_path: Optional custom output path
            
        Returns:
            Path to exported JSON file
        """
        try:
            if not output_path:
                video_name = Path(video_data.get('file_path', 'unknown')).stem
                output_path = self.output_dir / f"{video_name}_summary.json"
            
            # Prepare export data
            export_data = {
                'video_info': {
                    'file_path': video_data.get('file_path'),
                    'title': video_data.get('title', 'Unknown'),
                    'duration': video_data.get('duration', 0),
                    'total_scenes': video_data.get('total_scenes', 0),
                    'processed_date': datetime.now().isoformat()
                },
                'scenes': video_data.get('scenes', []),
                'transcription': video_data.get('transcription', []),
                'narrative_analysis': video_data.get('narrative_analysis', {}),
                'summary': video_data.get('summary', {}),
                'validation_metrics': video_data.get('validation_metrics', {})
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"JSON summary exported: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to export JSON summary: {e}")
            raise
    
    def export_vlc_bookmarks(self, video_data: Dict, output_path: Optional[str] = None) -> str:
        """
        Export VLC bookmark playlist (XSPF format).
        
        Args:
            video_data: Video analysis data with key moments
            output_path: Optional custom output path
            
        Returns:
            Path to exported XSPF file
        """
        try:
            if not output_path:
                video_name = Path(video_data.get('file_path', 'unknown')).stem
                output_path = self.output_dir / f"{video_name}_bookmarks.xspf"
            
            # Create XSPF structure
            root = ET.Element('playlist')
            root.set('version', '1')
            root.set('xmlns', 'http://xspf.org/ns/0/')
            
            # Add title
            title_elem = ET.SubElement(root, 'title')
            title_elem.text = f"Bookmarks - {video_data.get('title', 'Unknown Video')}"
            
            # Add track list
            tracklist = ET.SubElement(root, 'trackList')
            
            # Get key moments from narrative analysis
            narrative_analysis = video_data.get('narrative_analysis', {})
            key_moments = narrative_analysis.get('key_moments', [])
            
            video_path = video_data.get('file_path', '')
            
            for moment in key_moments:
                track = ET.SubElement(tracklist, 'track')
                
                # Track location
                location = ET.SubElement(track, 'location')
                location.text = f"file://{video_path}"
                
                # Track title
                track_title = ET.SubElement(track, 'title')
                track_title.text = moment.get('description', 'Key Moment')
                
                # Add extension for timestamp
                extension = ET.SubElement(track, 'extension')
                extension.set('application', 'http://www.videolan.org/vlc/playlist/ns/0/')
                
                # VLC option for start time
                vlc_option = ET.SubElement(extension, 'vlc:option')
                timestamp = moment.get('timestamp', 0)
                vlc_option.text = f"start-time={int(timestamp)}"
            
            # Write XSPF file
            tree = ET.ElementTree(root)
            tree.write(output_path, encoding='utf-8', xml_declaration=True)
            
            logger.info(f"VLC bookmarks exported: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to export VLC bookmarks: {e}")
            raise
    
    def export_summary_report(self, video_data: Dict, output_path: Optional[str] = None) -> str:
        """
        Export human-readable summary report as markdown.
        
        Args:
            video_data: Video analysis data
            output_path: Optional custom output path
            
        Returns:
            Path to exported markdown file
        """
        try:
            if not output_path:
                video_name = Path(video_data.get('file_path', 'unknown')).stem
                output_path = self.output_dir / f"{video_name}_report.md"
            
            # Generate markdown report
            report_lines = []
            
            # Header
            title = video_data.get('title', 'Unknown Video')
            report_lines.append(f"# Video Summary Report: {title}")
            report_lines.append("")
            
            # Video Information
            report_lines.append("## Video Information")
            report_lines.append(f"- **File**: {video_data.get('file_path', 'Unknown')}")
            report_lines.append(f"- **Duration**: {video_data.get('duration', 0)/60:.1f} minutes")
            report_lines.append(f"- **Total Scenes**: {video_data.get('total_scenes', 0)}")
            report_lines.append(f"- **Processed**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append("")
            
            # Narrative Analysis
            narrative_analysis = video_data.get('narrative_analysis', {})
            if narrative_analysis:
                report_lines.append("## Narrative Analysis")
                
                structure = narrative_analysis.get('structure_analysis', {})
                if structure:
                    report_lines.append(f"- **Structure**: {structure.get('act_structure', 'Unknown')}")
                    report_lines.append(f"- **Pacing**: {structure.get('pacing', 'Unknown')}")
                    report_lines.append(f"- **Tone**: {structure.get('tone', 'Unknown')}")
                
                # Key moments
                key_moments = narrative_analysis.get('key_moments', [])
                if key_moments:
                    report_lines.append("\n### Key Moments")
                    for i, moment in enumerate(key_moments[:10], 1):
                        timestamp = moment.get('timestamp', 0)
                        minutes, seconds = divmod(int(timestamp), 60)
                        description = moment.get('description', 'Key moment')
                        report_lines.append(f"{i}. **{minutes:02d}:{seconds:02d}** - {description}")
                
                report_lines.append("")
            
            # Validation Metrics
            validation = video_data.get('validation_metrics', {})
            if validation:
                report_lines.append("## Quality Metrics")
                f1_score = validation.get('f1_score', 0)
                precision = validation.get('precision', 0)
                recall = validation.get('recall', 0)
                report_lines.append(f"- **F1 Score**: {f1_score:.3f}")
                report_lines.append(f"- **Precision**: {precision:.3f}")
                report_lines.append(f"- **Recall**: {recall:.3f}")
                report_lines.append("")
            
            # Summary
            summary = video_data.get('summary', {})
            if summary:
                report_lines.append("## Summary")
                summary_text = summary.get('plot_summary', 'No summary available')
                report_lines.append(summary_text)
                report_lines.append("")
            
            # Write report
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))
            
            logger.info(f"Summary report exported: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to export summary report: {e}")
            raise
    
    def export_all_formats(self, video_data: Dict, base_name: Optional[str] = None) -> Dict[str, str]:
        """
        Export video summary in all available formats.
        
        Args:
            video_data: Complete video analysis data
            base_name: Optional base name for output files
            
        Returns:
            Dictionary mapping format names to file paths
        """
        if not base_name:
            base_name = Path(video_data.get('file_path', 'unknown')).stem
        
        exports = {}
        
        try:
            # JSON export
            json_path = self.output_dir / f"{base_name}_summary.json"
            exports['json'] = self.export_summary_json(video_data, json_path)
            
            # VLC bookmarks
            xspf_path = self.output_dir / f"{base_name}_bookmarks.xspf"
            exports['vlc_bookmarks'] = self.export_vlc_bookmarks(video_data, xspf_path)
            
            # Markdown report
            md_path = self.output_dir / f"{base_name}_report.md"
            exports['report'] = self.export_summary_report(video_data, md_path)
            
            logger.info(f"All formats exported for: {base_name}")
            return exports
            
        except Exception as e:
            logger.error(f"Failed to export all formats: {e}")
            raise