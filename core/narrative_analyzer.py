"""
Narrative analysis module using local LLM for offline film structure understanding.
Adapted from coursequery's LLM integration for movie analysis instead of RAG queries.
"""
import torch
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
import re

try:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, 
        pipeline, GenerationConfig
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    # Create dummy classes to prevent import errors
    class DummyTokenizer:
        def __init__(self):
            self.eos_token_id = 2
        
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return DummyTokenizer()
        
        def __call__(self, *args, **kwargs):
            return {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}
    
    class DummyModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return DummyModel()
        
        def parameters(self):
            return []
    
    class DummyPipeline:
        def __init__(self, *args, **kwargs):
            pass
        
        def __call__(self, *args, **kwargs):
            return [{"generated_text": "Dummy analysis - transformers not available"}]
    
    class DummyGenerationConfig:
        pass
    
    # Assign dummy classes to the expected names
    AutoTokenizer = DummyTokenizer
    AutoModelForCausalLM = DummyModel
    GenerationConfig = DummyGenerationConfig
    
    def pipeline(*args, **kwargs):
        return DummyPipeline(*args, **kwargs)

from config_detector import CONFIG as config, ENVIRONMENT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NarrativeAnalyzer:
    """Local LLM-powered narrative analysis for movies."""
    
    def __init__(self):
        """Initialize narrative analyzer with local LLM."""
        if ENVIRONMENT == "rtx_3060_local":
            self.device = torch.device(config.CUDA_DEVICE)  # RTX 3060
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")  # Generic CUDA
        else:
            self.device = torch.device("cpu")  # CPU fallback
        self.model = None
        self.tokenizer = None
        self.generator = None
        self.max_tokens = config.LLM_MAX_TOKENS
        
        logger.info(f"Narrative analyzer initialized on device: {self.device}")
    
    def load_model(self, model_path: str = None) -> bool:
        """
        Load local LLM model following coursequery pattern.
        
        Args:
            model_path: Path to local model directory
            
        Returns:
            True if model loaded successfully
        """
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers library not available. LLM functionality disabled.")
            logger.info("To enable LLM features, please install transformers: pip install transformers")
            return False
            
        try:
            model_path = model_path or str(config.LLM_MODEL_PATH)
            
            if not Path(model_path).exists():
                logger.error(f"Model path does not exist: {model_path}")
                logger.info("Please place your local LLM model in the 'models/local_llm' directory")
                return False
            
            logger.info(f"Loading local LLM from: {model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True
            )
            
            # Load model with GPU optimization
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                device_map="auto" if self.device.type == 'cuda' else None,
                trust_remote_code=True,
                local_files_only=True
            )
            
            # Create generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
            )
            
            logger.info("Local LLM loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load local LLM: {e}")
            return False
    
    def analyze_narrative_structure(self, transcription_data: List[Dict], 
                                  scenes: List[Dict]) -> Dict:
        """
        Analyze narrative structure of the movie using transcription and scene data.
        
        Args:
            transcription_data: List of transcription segments
            scenes: List of scene data
            
        Returns:
            Dictionary containing narrative analysis
        """
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available - providing basic structure analysis")
            return self._basic_narrative_analysis(transcription_data, scenes)
            
        if not self.model:
            if not self.load_model():
                logger.warning("Failed to load LLM - falling back to basic analysis")
                return self._basic_narrative_analysis(transcription_data, scenes)
        
        logger.info("Starting narrative structure analysis")
        
        try:
            # Prepare context from transcription and scenes
            context = self._prepare_narrative_context(transcription_data, scenes)
            
            # Analyze story structure
            structure_analysis = self._analyze_story_structure(context)
            
            # Identify key narrative moments
            key_moments = self._identify_key_moments(context, scenes)
            
            # Analyze character arcs and themes
            character_analysis = self._analyze_characters_and_themes(context)
            
            # Generate narrative summary
            narrative_summary = self._generate_narrative_summary(context)
            
            analysis_result = {
                'structure_analysis': structure_analysis,
                'key_moments': key_moments,
                'character_analysis': character_analysis,
                'narrative_summary': narrative_summary,
                'total_scenes': len(scenes),
                'total_duration': scenes[-1]['end_time'] if scenes else 0
            }
            
            logger.info("Narrative analysis completed")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Narrative analysis failed: {e}")
            logger.warning("Falling back to basic analysis")
            return self._basic_narrative_analysis(transcription_data, scenes)
    
    def _prepare_narrative_context(self, transcription_data: List[Dict], 
                                 scenes: List[Dict]) -> str:
        """Prepare context text for LLM analysis."""
        context_parts = []
        
        # Add movie overview
        context_parts.append("MOVIE ANALYSIS CONTEXT:")
        context_parts.append(f"Total Scenes: {len(scenes)}")
        
        if transcription_data:
            total_duration = transcription_data[-1]['end_time']
            context_parts.append(f"Duration: {total_duration/60:.1f} minutes")
        
        context_parts.append("\nSCENE BREAKDOWN:")
        
        # Combine scenes with their transcription
        for scene in scenes[:20]:  # Limit to first 20 scenes for context
            scene_text = self._get_scene_transcription(scene, transcription_data)
            
            context_parts.append(
                f"Scene {scene['scene_number']} "
                f"({scene['start_time']:.1f}s-{scene['end_time']:.1f}s): "
                f"{scene_text[:200]}..."
            )
        
        return "\n".join(context_parts)
    
    def _get_scene_transcription(self, scene: Dict, transcription_data: List[Dict]) -> str:
        """Extract transcription text for a specific scene."""
        scene_start = scene['start_time']
        scene_end = scene['end_time']
        
        scene_text = []
        for trans in transcription_data:
            if (trans['start_time'] >= scene_start and trans['end_time'] <= scene_end):
                scene_text.append(trans['text'])
        
        return ' '.join(scene_text) if scene_text else "[No dialogue]"
    
    def _analyze_story_structure(self, context: str) -> Dict:
        """Analyze the story structure using LLM."""
        prompt = f"""
        Analyze the following movie's narrative structure. Identify the three-act structure:
        
        {context[:2000]}
        
        Please identify:
        1. Act 1 (Setup): Key scenes and duration
        2. Act 2 (Confrontation): Major conflicts and turning points
        3. Act 3 (Resolution): Climax and resolution
        4. Key plot points and their approximate timestamps
        
        Structure Analysis:"""
        
        try:
            response = self.generator(
                prompt,
                max_new_tokens=500,
                temperature=config.LLM_TEMPERATURE,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            analysis_text = response[0]['generated_text'][len(prompt):].strip()
            
            # Parse the response into structured data
            return self._parse_structure_analysis(analysis_text)
            
        except Exception as e:
            logger.error(f"Story structure analysis failed: {e}")
            return {'error': str(e)}
    
    def _identify_key_moments(self, context: str, scenes: List[Dict]) -> List[Dict]:
        """Identify key narrative moments for summary inclusion."""
        prompt = f"""
        From this movie analysis, identify the 5-10 most important scenes for a summary:
        
        {context[:2000]}
        
        For each key moment, provide:
        - Scene number or timestamp
        - Brief description (1-2 sentences)
        - Importance reason
        
        Key Moments:"""
        
        try:
            response = self.generator(
                prompt,
                max_new_tokens=400,
                temperature=config.LLM_TEMPERATURE,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            moments_text = response[0]['generated_text'][len(prompt):].strip()
            
            # Parse and match to actual scenes
            return self._parse_key_moments(moments_text, scenes)
            
        except Exception as e:
            logger.error(f"Key moments identification failed: {e}")
            return []
    
    def _analyze_characters_and_themes(self, context: str) -> Dict:
        """Analyze main characters and themes."""
        prompt = f"""
        Analyze the main characters and themes in this movie:
        
        {context[:2000]}
        
        Provide:
        1. Main characters and their roles
        2. Character arcs and development
        3. Major themes and messages
        4. Tone and genre elements
        
        Analysis:"""
        
        try:
            response = self.generator(
                prompt,
                max_new_tokens=400,
                temperature=config.LLM_TEMPERATURE,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            analysis_text = response[0]['generated_text'][len(prompt):].strip()
            
            return {
                'raw_analysis': analysis_text,
                'characters': self._extract_characters(analysis_text),
                'themes': self._extract_themes(analysis_text)
            }
            
        except Exception as e:
            logger.error(f"Character and theme analysis failed: {e}")
            return {'error': str(e)}
    
    def _generate_narrative_summary(self, context: str) -> str:
        """Generate a concise narrative summary."""
        prompt = f"""
        Write a concise 3-paragraph summary of this movie's plot:
        
        {context[:2000]}
        
        Summary:"""
        
        try:
            response = self.generator(
                prompt,
                max_new_tokens=300,
                temperature=config.LLM_TEMPERATURE,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            return response[0]['generated_text'][len(prompt):].strip()
            
        except Exception as e:
            logger.error(f"Narrative summary generation failed: {e}")
            return f"Summary generation failed: {str(e)}"
    
    def _parse_structure_analysis(self, analysis_text: str) -> Dict:
        """Parse LLM structure analysis into structured data."""
        structure = {
            'act1': {'description': '', 'duration': ''},
            'act2': {'description': '', 'duration': ''},
            'act3': {'description': '', 'duration': ''},
            'plot_points': []
        }
        
        # Simple regex-based parsing
        act1_match = re.search(r'Act 1[:\s]*([^\n]+)', analysis_text, re.IGNORECASE)
        if act1_match:
            structure['act1']['description'] = act1_match.group(1).strip()
        
        act2_match = re.search(r'Act 2[:\s]*([^\n]+)', analysis_text, re.IGNORECASE)
        if act2_match:
            structure['act2']['description'] = act2_match.group(1).strip()
        
        act3_match = re.search(r'Act 3[:\s]*([^\n]+)', analysis_text, re.IGNORECASE)
        if act3_match:
            structure['act3']['description'] = act3_match.group(1).strip()
        
        return structure
    
    def _parse_key_moments(self, moments_text: str, scenes: List[Dict]) -> List[Dict]:
        """Parse key moments and match to scene data."""
        key_moments = []
        
        # Extract scene numbers or timestamps from the text
        lines = moments_text.split('\n')
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['scene', 'moment', 'timestamp']):
                # Try to extract scene number or time
                scene_match = re.search(r'scene\s+(\d+)', line, re.IGNORECASE)
                time_match = re.search(r'(\d+):(\d+)', line)
                
                moment = {
                    'description': line.strip(),
                    'scene_number': None,
                    'timestamp': None,
                    'importance': 'high'
                }
                
                if scene_match:
                    scene_num = int(scene_match.group(1))
                    if scene_num <= len(scenes):
                        moment['scene_number'] = scene_num
                        moment['timestamp'] = scenes[scene_num - 1]['start_time']
                
                elif time_match:
                    minutes = int(time_match.group(1))
                    seconds = int(time_match.group(2))
                    moment['timestamp'] = minutes * 60 + seconds
                
                key_moments.append(moment)
        
        return key_moments[:10]  # Limit to 10 key moments
    
    def _extract_characters(self, analysis_text: str) -> List[str]:
        """Extract character names from analysis text."""
        # Simple extraction - look for capitalized names
        characters = []
        
        lines = analysis_text.split('\n')
        for line in lines:
            if 'character' in line.lower():
                # Extract capitalized words that might be names
                words = line.split()
                for word in words:
                    if (word[0].isupper() and len(word) > 2 and 
                        word not in ['The', 'A', 'An', 'Character', 'Main', 'Character']):
                        if word not in characters:
                            characters.append(word)
        
        return characters[:5]  # Limit to 5 main characters
    
    def _extract_themes(self, analysis_text: str) -> List[str]:
        """Extract themes from analysis text."""
        common_themes = [
            'love', 'betrayal', 'revenge', 'redemption', 'friendship',
            'family', 'justice', 'sacrifice', 'coming of age', 'survival',
            'power', 'corruption', 'identity', 'freedom', 'destiny'
        ]
        
        themes = []
        analysis_lower = analysis_text.lower()
        
        for theme in common_themes:
            if theme in analysis_lower:
                themes.append(theme.title())
        
        return themes[:3]  # Limit to 3 main themes
    
    def calculate_scene_narrative_importance(self, scenes: List[Dict], 
                                           key_moments: List[Dict]) -> List[Dict]:
        """Calculate narrative importance scores for scenes based on LLM analysis."""
        logger.info("Calculating narrative importance scores")
        
        for scene in scenes:
            base_score = scene.get('importance_score', 0.5)
            narrative_bonus = 0.0
            
            # Check if scene matches any key moments
            for moment in key_moments:
                if (moment.get('scene_number') == scene['scene_number'] or
                    (moment.get('timestamp') and 
                     scene['start_time'] <= moment['timestamp'] <= scene['end_time'])):
                    narrative_bonus += 0.3
            
            # Position-based narrative importance
            total_scenes = len(scenes)
            position = scene['scene_number']
            
            # Opening scenes (setup)
            if position <= total_scenes * 0.1:
                narrative_bonus += 0.2
            # Middle scenes (confrontation)
            elif 0.4 <= position/total_scenes <= 0.7:
                narrative_bonus += 0.15
            # Climax and resolution
            elif position >= total_scenes * 0.8:
                narrative_bonus += 0.25
            
            # Combine scores
            scene['narrative_importance'] = min(base_score + narrative_bonus, 1.0)
        
        return scenes
    
    def _basic_narrative_analysis(self, transcription_data: List[Dict], 
                                scenes: List[Dict]) -> Dict:
        """
        Provide basic narrative analysis when transformers is not available.
        Uses rule-based heuristics instead of LLM analysis.
        """
        logger.info("Using basic narrative analysis (no LLM)")
        
        total_duration = scenes[-1]['end_time'] if scenes else 0
        total_scenes = len(scenes)
        
        # Basic 5-act structure assignment
        key_moments = []
        if total_scenes > 0:
            act_boundaries = [
                int(total_scenes * 0.1),   # Act 1 end
                int(total_scenes * 0.25),  # Act 2a end
                int(total_scenes * 0.5),   # Act 2b end
                int(total_scenes * 0.75),  # Act 3 end
                total_scenes - 1           # Act 4 end
            ]
            
            key_moments = [
                {
                    'type': 'opening',
                    'description': 'Opening sequence',
                    'scene_number': 1,
                    'timestamp': scenes[0]['start_time'] if scenes else 0,
                    'importance': 0.8
                },
                {
                    'type': 'inciting_incident',
                    'description': 'Inciting incident (estimated)',
                    'scene_number': act_boundaries[0],
                    'timestamp': scenes[act_boundaries[0]]['start_time'] if act_boundaries[0] < len(scenes) else 0,
                    'importance': 0.9
                },
                {
                    'type': 'midpoint',
                    'description': 'Story midpoint',
                    'scene_number': act_boundaries[2],
                    'timestamp': scenes[act_boundaries[2]]['start_time'] if act_boundaries[2] < len(scenes) else 0,
                    'importance': 0.85
                },
                {
                    'type': 'climax',
                    'description': 'Climax (estimated)',
                    'scene_number': act_boundaries[3],
                    'timestamp': scenes[act_boundaries[3]]['start_time'] if act_boundaries[3] < len(scenes) else 0,
                    'importance': 1.0
                }
            ]
        
        return {
            'structure_analysis': {
                'act_structure': '5-act structure (estimated)',
                'pacing': 'Standard narrative pacing',
                'tone': 'Unable to analyze without LLM',
                'genre_indicators': []
            },
            'key_moments': key_moments,
            'character_analysis': {
                'main_characters': [],
                'character_arcs': [],
                'themes': ['Basic analysis mode - install transformers for detailed analysis']
            },
            'narrative_summary': {
                'plot_summary': 'Narrative analysis requires transformers library for detailed insights',
                'structure_notes': f'Movie has {total_scenes} scenes over {total_duration/60:.1f} minutes'
            },
            'total_scenes': total_scenes,
            'total_duration': total_duration
        }

    def get_model_info(self) -> Dict:
        """Get information about the loaded LLM."""
        if not TRANSFORMERS_AVAILABLE:
            return {"status": "Transformers library not available"}
            
        if not self.model:
            return {"status": "No model loaded"}
        
        return {
            "model_path": str(config.LLM_MODEL_PATH),
            "device": str(self.device),
            "parameters": sum(p.numel() for p in self.model.parameters()) if self.model else 0,
            "cuda_available": torch.cuda.is_available(),
            "gpu_memory_allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        }
