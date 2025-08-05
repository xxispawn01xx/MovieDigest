"""
Plex Media Server integration for enhanced video discovery and metadata.
Provides genre filtering, ratings, and rich metadata from Plex libraries.
"""
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus
import json

logger = logging.getLogger(__name__)

class PlexIntegration:
    """Integrates with Plex Media Server for enhanced video discovery."""
    
    def __init__(self, server_url: str = None, token: str = None):
        """
        Initialize Plex integration.
        
        Args:
            server_url: Plex server URL (e.g., 'http://localhost:32400')
            token: Plex authentication token
        """
        self.server_url = server_url
        self.token = token
        self.session = requests.Session()
        self.libraries = {}
        self.connection_verified = False
        
        if self.server_url and self.token:
            self.verify_connection()
    
    def verify_connection(self) -> bool:
        """
        Verify connection to Plex server.
        
        Returns:
            True if connection successful
        """
        try:
            url = f"{self.server_url}/identity"
            headers = {'X-Plex-Token': self.token}
            
            response = self.session.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            self.connection_verified = True
            logger.info("Plex connection verified successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Plex server: {e}")
            self.connection_verified = False
            return False
    
    def get_libraries(self) -> Dict[str, Dict]:
        """
        Get all Plex libraries.
        
        Returns:
            Dictionary of library information
        """
        if not self.connection_verified:
            return {}
        
        try:
            url = f"{self.server_url}/library/sections"
            headers = {'X-Plex-Token': self.token}
            
            response = self.session.get(url, headers=headers)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            libraries = {}
            
            for directory in root.findall('Directory'):
                lib_id = directory.get('key')
                lib_name = directory.get('title')
                lib_type = directory.get('type')
                
                if lib_type in ['movie', 'show']:
                    libraries[lib_id] = {
                        'name': lib_name,
                        'type': lib_type,
                        'key': lib_id
                    }
            
            self.libraries = libraries
            logger.info(f"Found {len(libraries)} media libraries")
            return libraries
            
        except Exception as e:
            logger.error(f"Failed to get Plex libraries: {e}")
            return {}
    
    def get_movies_by_genre(self, genre: str, library_key: str = None) -> List[Dict]:
        """
        Get movies filtered by genre.
        
        Args:
            genre: Genre to filter by
            library_key: Specific library to search (optional)
            
        Returns:
            List of movie metadata
        """
        if not self.connection_verified:
            return []
        
        try:
            # If no library specified, search all movie libraries
            if not library_key:
                movie_libraries = [
                    lib for lib in self.libraries.values() 
                    if lib['type'] == 'movie'
                ]
                if not movie_libraries:
                    return []
                library_key = movie_libraries[0]['key']
            
            # Search for movies by genre
            url = f"{self.server_url}/library/sections/{library_key}/all"
            headers = {'X-Plex-Token': self.token}
            params = {'genre': genre}
            
            response = self.session.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            movies = []
            
            for video in root.findall('Video'):
                movie_data = self._parse_movie_metadata(video)
                if movie_data:
                    movies.append(movie_data)
            
            logger.info(f"Found {len(movies)} movies in genre: {genre}")
            return movies
            
        except Exception as e:
            logger.error(f"Failed to get movies by genre: {e}")
            return []
    
    def search_movies(self, query: str, filters: Dict = None) -> List[Dict]:
        """
        Search movies with optional filters.
        
        Args:
            query: Search query
            filters: Additional filters (genre, year, rating, etc.)
            
        Returns:
            List of matching movies
        """
        if not self.connection_verified:
            return []
        
        try:
            movie_libraries = [
                lib for lib in self.libraries.values() 
                if lib['type'] == 'movie'
            ]
            
            all_movies = []
            
            for library in movie_libraries:
                url = f"{self.server_url}/library/sections/{library['key']}/search"
                headers = {'X-Plex-Token': self.token}
                params = {'query': query}
                
                # Add filters if provided
                if filters:
                    if 'genre' in filters:
                        params['genre'] = filters['genre']
                    if 'year' in filters:
                        params['year'] = filters['year']
                    if 'contentRating' in filters:
                        params['contentRating'] = filters['contentRating']
                
                response = self.session.get(url, headers=headers, params=params)
                response.raise_for_status()
                
                root = ET.fromstring(response.content)
                
                for video in root.findall('Video'):
                    movie_data = self._parse_movie_metadata(video)
                    if movie_data:
                        all_movies.append(movie_data)
            
            return all_movies
            
        except Exception as e:
            logger.error(f"Failed to search movies: {e}")
            return []
    
    def get_movie_details(self, movie_key: str) -> Optional[Dict]:
        """
        Get detailed information for a specific movie.
        
        Args:
            movie_key: Plex movie key/ID
            
        Returns:
            Detailed movie metadata
        """
        if not self.connection_verified:
            return None
        
        try:
            url = f"{self.server_url}{movie_key}"
            headers = {'X-Plex-Token': self.token}
            
            response = self.session.get(url, headers=headers)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            video = root.find('Video')
            
            if video is not None:
                return self._parse_detailed_metadata(video)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get movie details: {e}")
            return None
    
    def get_genres(self, library_key: str = None) -> List[str]:
        """
        Get all available genres.
        
        Args:
            library_key: Specific library to get genres from
            
        Returns:
            List of genre names
        """
        if not self.connection_verified:
            return []
        
        try:
            if not library_key:
                movie_libraries = [
                    lib for lib in self.libraries.values() 
                    if lib['type'] == 'movie'
                ]
                if not movie_libraries:
                    return []
                library_key = movie_libraries[0]['key']
            
            url = f"{self.server_url}/library/sections/{library_key}/genre"
            headers = {'X-Plex-Token': self.token}
            
            response = self.session.get(url, headers=headers)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            genres = []
            
            for directory in root.findall('Directory'):
                genre_name = directory.get('title')
                if genre_name:
                    genres.append(genre_name)
            
            return sorted(genres)
            
        except Exception as e:
            logger.error(f"Failed to get genres: {e}")
            return []
    
    def _parse_movie_metadata(self, video_element) -> Optional[Dict]:
        """Parse basic movie metadata from Plex XML."""
        try:
            # Get media file path
            media_element = video_element.find('Media')
            part_element = media_element.find('Part') if media_element is not None else None
            file_path = part_element.get('file') if part_element is not None else None
            
            if not file_path:
                return None
            
            # Extract genres
            genres = []
            for genre in video_element.findall('Genre'):
                genre_name = genre.get('tag')
                if genre_name:
                    genres.append(genre_name)
            
            # Extract basic metadata
            movie_data = {
                'plex_key': video_element.get('key'),
                'title': video_element.get('title'),
                'file_path': file_path,
                'year': video_element.get('year'),
                'duration': int(video_element.get('duration', 0)) // 1000,  # Convert to seconds
                'rating': float(video_element.get('rating', 0)),
                'content_rating': video_element.get('contentRating'),
                'genres': genres,
                'summary': video_element.get('summary', ''),
                'studio': video_element.get('studio'),
                'added_date': video_element.get('addedAt'),
                'updated_date': video_element.get('updatedAt')
            }
            
            return movie_data
            
        except Exception as e:
            logger.error(f"Failed to parse movie metadata: {e}")
            return None
    
    def _parse_detailed_metadata(self, video_element) -> Dict:
        """Parse detailed movie metadata including cast, crew, etc."""
        basic_data = self._parse_movie_metadata(video_element)
        if not basic_data:
            return {}
        
        try:
            # Extract cast
            cast = []
            for role in video_element.findall('Role'):
                actor_name = role.get('tag')
                character = role.get('role')
                if actor_name:
                    cast.append({
                        'actor': actor_name,
                        'character': character or 'Unknown'
                    })
            
            # Extract directors
            directors = []
            for director in video_element.findall('Director'):
                director_name = director.get('tag')
                if director_name:
                    directors.append(director_name)
            
            # Extract writers
            writers = []
            for writer in video_element.findall('Writer'):
                writer_name = writer.get('tag')
                if writer_name:
                    writers.append(writer_name)
            
            # Media information
            media_element = video_element.find('Media')
            media_info = {}
            if media_element is not None:
                media_info = {
                    'resolution': media_element.get('videoResolution'),
                    'bitrate': media_element.get('bitrate'),
                    'container': media_element.get('container'),
                    'video_codec': media_element.get('videoCodec'),
                    'audio_codec': media_element.get('audioCodec')
                }
            
            # Combine all metadata
            detailed_data = {
                **basic_data,
                'cast': cast[:10],  # Limit to top 10 actors
                'directors': directors,
                'writers': writers,
                'media_info': media_info
            }
            
            return detailed_data
            
        except Exception as e:
            logger.error(f"Failed to parse detailed metadata: {e}")
            return basic_data
    
    def filter_movies_for_processing(self, criteria: Dict) -> List[Dict]:
        """
        Filter movies based on processing criteria.
        
        Args:
            criteria: Filter criteria (min_duration, genres, ratings, etc.)
            
        Returns:
            List of movies matching criteria
        """
        if not self.connection_verified:
            return []
        
        try:
            movie_libraries = [
                lib for lib in self.libraries.values() 
                if lib['type'] == 'movie'
            ]
            
            filtered_movies = []
            
            for library in movie_libraries:
                url = f"{self.server_url}/library/sections/{library['key']}/all"
                headers = {'X-Plex-Token': self.token}
                
                response = self.session.get(url, headers=headers)
                response.raise_for_status()
                
                root = ET.fromstring(response.content)
                
                for video in root.findall('Video'):
                    movie_data = self._parse_movie_metadata(video)
                    if movie_data and self._meets_criteria(movie_data, criteria):
                        filtered_movies.append(movie_data)
            
            return filtered_movies
            
        except Exception as e:
            logger.error(f"Failed to filter movies: {e}")
            return []
    
    def _meets_criteria(self, movie_data: Dict, criteria: Dict) -> bool:
        """Check if movie meets filtering criteria."""
        try:
            # Check minimum duration
            if 'min_duration_minutes' in criteria:
                min_duration = criteria['min_duration_minutes'] * 60
                if movie_data.get('duration', 0) < min_duration:
                    return False
            
            # Check maximum duration
            if 'max_duration_minutes' in criteria:
                max_duration = criteria['max_duration_minutes'] * 60
                if movie_data.get('duration', 0) > max_duration:
                    return False
            
            # Check genres
            if 'required_genres' in criteria:
                movie_genres = movie_data.get('genres', [])
                required_genres = criteria['required_genres']
                if not any(genre in movie_genres for genre in required_genres):
                    return False
            
            # Check excluded genres
            if 'excluded_genres' in criteria:
                movie_genres = movie_data.get('genres', [])
                excluded_genres = criteria['excluded_genres']
                if any(genre in movie_genres for genre in excluded_genres):
                    return False
            
            # Check minimum rating
            if 'min_rating' in criteria:
                movie_rating = movie_data.get('rating', 0)
                if movie_rating < criteria['min_rating']:
                    return False
            
            # Check year range
            if 'min_year' in criteria:
                movie_year = int(movie_data.get('year', 0))
                if movie_year < criteria['min_year']:
                    return False
            
            if 'max_year' in criteria:
                movie_year = int(movie_data.get('year', 0))
                if movie_year > criteria['max_year']:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking criteria: {e}")
            return False
    
    def get_connection_status(self) -> Dict:
        """Get current connection status and server info."""
        status = {
            'connected': self.connection_verified,
            'server_url': self.server_url,
            'has_token': bool(self.token),
            'libraries_count': len(self.libraries),
            'movie_libraries': [
                lib for lib in self.libraries.values() 
                if lib['type'] == 'movie'
            ]
        }
        
        return status