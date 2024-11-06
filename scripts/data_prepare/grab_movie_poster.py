"""Movie poster grabber.

This module provides functionality to download movie poster images from online sources.
"""

import re
import os
import requests
import pandas as pd
import time

from typing import Optional, Tuple
from tqdm import tqdm


class MoviePosterGrabber:
    """Class to grab movie poster images from online sources."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the MoviePosterGrabber.
        
        Args:
            api_key: TMDB API key. If not provided, will try to get from environment variable TMDB_API_KEY
        """
        self.api_key = api_key or os.getenv("TMDB_API_KEY")
        if not self.api_key:
            raise ValueError("TMDB API key must be provided or set in TMDB_API_KEY environment variable")
        
        self.base_url = "https://api.themoviedb.org/3"
        self.image_base_url = "https://image.tmdb.org/t/p/original"

    def get_poster_url(self, movie_name: str) -> Tuple[Optional[str], bool]:
        """Get the poster URL for a given movie name.
        
        Args:
            movie_name: Name of the movie to search for
            
        Returns:
            Tuple of (URL of the movie poster if found, whether timeout occurred)
            URL will be None if not found or error occurred
        """
        search_url = f"{self.base_url}/search/movie"
        params = {
            "api_key": self.api_key,
            "query": movie_name
        }
        try:
            response = requests.get(search_url, params=params, timeout=5)
        except requests.exceptions.Timeout:
            return None, True
        except OSError:
            return None, False

        if response.status_code != 200:
            return None, False
            
        results = response.json().get("results", [])
        if not results:
            return None, False
            
        poster_path = results[0].get("poster_path")
        if not poster_path:
            return None, False
            
        return f"{self.image_base_url}{poster_path}", False

    def download_poster(self, movie_name: str, save_path: str) -> Tuple[bool, bool]:
        """Download the movie poster and save it to the specified path.
        
        Args:
            movie_name: Name of the movie
            save_path: Path where the poster should be saved
            
        Returns:
            Tuple of (success status, whether timeout occurred)
        """
        poster_url, timeout = self.get_poster_url(movie_name)
        if timeout:
            return False, True
        if not poster_url:
            return False, False
        
        try:
            response = requests.get(poster_url, timeout=5)
        except requests.exceptions.Timeout:
            return False, True
        except OSError:
            return False, False

        if response.status_code != 200:
            return False, False
            
        with open(save_path, "wb") as f:
            f.write(response.content)
            
        return True, False


if __name__ == "__main__":
    data = pd.read_csv("~/Datasets/MovieLens20M/movie.csv")
    movie_poster_grabber = MoviePosterGrabber()
    ROOT_DIR = "/home/limjk/Datasets/MovieLens20M/posters"
    wait_time = 3.0

    for _, row in tqdm(data.iterrows(), total=data.shape[0], desc="Grabbing posters"):
        id, title = row["movieId"], row["title"]
        # Remove year from title using regex (e.g., "Movie Title (1999)" -> "Movie Title")
        title = re.sub(r'\s*\(\d{4}\)\s*$', '', title)

        forbidden_chars = [' ', '/', ':', '?', '*', '<', '>', '|', '\\', '"', ',', '.']
        striped_movie_name = title
        for char in forbidden_chars:
            striped_movie_name = striped_movie_name.replace(char, '_')

        striped_movie_name = striped_movie_name.replace("__", "_")
        striped_movie_name = striped_movie_name.strip()

        file_name = f"{id}_{striped_movie_name}.jpg"
        save_path = os.path.join(ROOT_DIR, file_name)
        
        # Check if file exists and has non-zero size
        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            continue
            
        success, timeout = movie_poster_grabber.download_poster(title, save_path)

        if success:
            wait_time = max(3.0, wait_time - 0.5)
        elif timeout:
            wait_time = min(10.0, wait_time + 1.0)

        time.sleep(wait_time)
