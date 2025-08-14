"""
Legal-specific video processing module for deposition and expert witness content.
Handles legal terminology, speaker identification, and compliance requirements.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class LegalSpeaker:
    """Represents a speaker in legal proceedings."""
    speaker_id: str
    role: str  # 'attorney', 'witness', 'court_reporter', 'judge'
    name: Optional[str] = None
    affiliation: Optional[str] = None  # Law firm, company, etc.

@dataclass
class LegalSegment:
    """Represents a segment of legal video with metadata."""
    start_time: float
    end_time: float
    speaker: LegalSpeaker
    content: str
    segment_type: str  # 'question', 'answer', 'objection', 'exhibit', 'break'
    key_terms: List[str]
    exhibits_referenced: List[str]

@dataclass
class Objection:
    """Represents a legal objection during proceedings."""
    timestamp: float
    objecting_attorney: str
    objection_type: str  # 'relevance', 'hearsay', 'foundation', etc.
    ruling: Optional[str] = None  # 'sustained', 'overruled'
    transcript_reference: Optional[str] = None

class LegalTerminologyProcessor:
    """Processes legal terminology and identifies key legal concepts."""
    
    def __init__(self):
        self.legal_terms = {
            'objections': [
                'objection', 'hearsay', 'relevance', 'foundation', 'speculation',
                'argumentative', 'compound', 'leading', 'assumes facts',
                'sustained', 'overruled'
            ],
            'evidence': [
                'exhibit', 'document', 'photograph', 'video', 'recording',
                'marked', 'identified', 'authenticated', 'foundation'
            ],
            'qualifications': [
                'education', 'experience', 'expert', 'opinion', 'qualified',
                'credentials', 'board certified', 'licensed'
            ],
            'legal_concepts': [
                'negligence', 'liability', 'damages', 'causation', 'breach',
                'standard of care', 'proximate cause', 'foreseeability'
            ]
        }
    
    def identify_legal_terms(self, text: str) -> Dict[str, List[str]]:
        """Identify legal terms in transcript text."""
        found_terms = {category: [] for category in self.legal_terms}
        
        for category, terms in self.legal_terms.items():
            for term in terms:
                if re.search(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE):
                    found_terms[category].append(term)
        
        return found_terms

class DepositionProcessor:
    """Main processor for deposition videos and transcripts."""
    
    def __init__(self):
        self.terminology_processor = LegalTerminologyProcessor()
        
    def process_deposition_video(self, video_path: str, transcript_path: Optional[str] = None) -> Dict:
        """
        Process a deposition video with optional transcript synchronization.
        
        Args:
            video_path: Path to the deposition video file
            transcript_path: Optional path to transcript file
            
        Returns:
            Dict containing processed deposition data
        """
        try:
            logger.info(f"Processing deposition video: {video_path}")
            
            # Initialize processing results
            results = {
                'video_path': video_path,
                'processed_at': datetime.now().isoformat(),
                'speakers': [],
                'segments': [],
                'objections': [],
                'exhibits': [],
                'key_testimony': [],
                'summary_metadata': {}
            }
            
            # Process transcript if available
            if transcript_path:
                results.update(self._process_transcript(transcript_path))
            
            # Identify speakers and roles
            results['speakers'] = self._identify_speakers(results.get('transcript_text', ''))
            
            # Detect objections and rulings
            results['objections'] = self._detect_objections(results.get('transcript_text', ''))
            
            # Identify exhibit references
            results['exhibits'] = self._identify_exhibits(results.get('transcript_text', ''))
            
            # Generate legal citation format
            results['citation_format'] = self._generate_citation_format(video_path)
            
            logger.info(f"Deposition processing completed with {len(results['objections'])} objections found")
            return results
            
        except Exception as e:
            logger.error(f"Error processing deposition video {video_path}: {e}")
            return {'error': str(e), 'video_path': video_path}
    
    def _process_transcript(self, transcript_path: str) -> Dict:
        """Process transcript file and extract text."""
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript_text = f.read()
            
            return {
                'transcript_path': transcript_path,
                'transcript_text': transcript_text,
                'transcript_length': len(transcript_text.split())
            }
        except Exception as e:
            logger.error(f"Error processing transcript {transcript_path}: {e}")
            return {}
    
    def _identify_speakers(self, transcript_text: str) -> List[LegalSpeaker]:
        """Identify speakers and their roles in the deposition."""
        speakers = []
        
        # Common deposition speaker patterns
        attorney_patterns = [
            r'(?i)attorney\s+for\s+(?:plaintiff|defendant)',
            r'(?i)mr\.\s+\w+:|ms\.\s+\w+:',
            r'(?i)counsel:'
        ]
        
        witness_patterns = [
            r'(?i)witness:|deponent:',
            r'(?i)the\s+witness:'
        ]
        
        # Extract speaker information (simplified implementation)
        # In production, this would be more sophisticated
        if re.search('|'.join(attorney_patterns), transcript_text):
            speakers.append(LegalSpeaker(
                speaker_id="attorney_1",
                role="attorney",
                name="Examining Attorney"
            ))
        
        if re.search('|'.join(witness_patterns), transcript_text):
            speakers.append(LegalSpeaker(
                speaker_id="witness_1", 
                role="witness",
                name="Deponent"
            ))
        
        return speakers
    
    def _detect_objections(self, transcript_text: str) -> List[Objection]:
        """Detect objections and rulings in transcript."""
        objections = []
        
        # Objection patterns
        objection_pattern = r'(?i)(objection[^.]*?(?:sustained|overruled))'
        matches = re.finditer(objection_pattern, transcript_text)
        
        for match in matches:
            objection_text = match.group(1)
            
            # Determine objection type
            objection_type = "general"
            if "hearsay" in objection_text.lower():
                objection_type = "hearsay"
            elif "relevance" in objection_text.lower():
                objection_type = "relevance"
            elif "foundation" in objection_text.lower():
                objection_type = "foundation"
            
            # Determine ruling
            ruling = None
            if "sustained" in objection_text.lower():
                ruling = "sustained"
            elif "overruled" in objection_text.lower():
                ruling = "overruled"
            
            objections.append(Objection(
                timestamp=0.0,  # Would need video sync for actual timestamp
                objecting_attorney="Attorney",
                objection_type=objection_type,
                ruling=ruling,
                transcript_reference=objection_text
            ))
        
        return objections
    
    def _identify_exhibits(self, transcript_text: str) -> List[Dict]:
        """Identify exhibit references in transcript."""
        exhibits = []
        
        # Exhibit patterns
        exhibit_patterns = [
            r'(?i)exhibit\s+([a-z0-9-]+)',
            r'(?i)marked\s+as\s+exhibit\s+([a-z0-9-]+)',
            r'(?i)showing\s+you\s+exhibit\s+([a-z0-9-]+)'
        ]
        
        for pattern in exhibit_patterns:
            matches = re.finditer(pattern, transcript_text)
            for match in matches:
                exhibit_id = match.group(1)
                if exhibit_id not in [e['id'] for e in exhibits]:
                    exhibits.append({
                        'id': exhibit_id,
                        'first_reference': match.start(),
                        'context': match.group(0)
                    })
        
        return exhibits
    
    def _generate_citation_format(self, video_path: str) -> str:
        """Generate legal citation format for the deposition."""
        import os
        filename = os.path.basename(video_path)
        
        # Extract case information from filename if available
        # Format: "Plaintiff Dep. Page:Line-Line"
        return f"{filename} Dep."

class ExpertWitnessProcessor:
    """Processor for expert witness testimony videos."""
    
    def __init__(self):
        self.qualification_keywords = [
            'education', 'degree', 'university', 'board certified',
            'licensed', 'experience', 'publications', 'cases',
            'testified', 'expert', 'opinion'
        ]
    
    def process_expert_testimony(self, video_path: str) -> Dict:
        """Process expert witness testimony video."""
        try:
            logger.info(f"Processing expert witness testimony: {video_path}")
            
            results = {
                'video_path': video_path,
                'expert_type': 'general',
                'qualifications_section': None,
                'opinion_sections': [],
                'credibility_factors': {},
                'processed_at': datetime.now().isoformat()
            }
            
            # In production, would extract actual expert qualifications
            # and opinion testimony sections
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing expert testimony {video_path}: {e}")
            return {'error': str(e), 'video_path': video_path}

# Integration with existing video processing pipeline
def process_legal_video(video_path: str, video_type: str = 'deposition') -> Dict:
    """
    Main entry point for legal video processing.
    
    Args:
        video_path: Path to the legal video file
        video_type: Type of legal video ('deposition', 'expert_witness', 'hearing')
        
    Returns:
        Dict containing processed legal video data
    """
    if video_type == 'deposition':
        processor = DepositionProcessor()
        return processor.process_deposition_video(video_path)
    elif video_type == 'expert_witness':
        processor = ExpertWitnessProcessor()
        return processor.process_expert_testimony(video_path)
    else:
        logger.warning(f"Unknown legal video type: {video_type}")
        return {'error': f'Unsupported video type: {video_type}'}