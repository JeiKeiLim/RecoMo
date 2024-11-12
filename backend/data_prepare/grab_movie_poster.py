"""Movie poster grabber.

This module provides functionality to download movie poster images from online sources.
"""

import os
import re
import time
from typing import Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm


class RateLimiter:
    """Handles rate limiting with dynamic backoff."""
    
    def __init__(self, initial_wait: float = 3.0, min_wait: float = 3.0, max_wait: float = 10.0):
        self.wait_time = initial_wait
        self.min_wait = min_wait
        self.max_wait = max_wait

    def success(self) -> None:
        """Decrease wait time after successful request."""
        self.wait_time = max(self.min_wait, self.wait_time - 0.5)

    def failure(self) -> None:
        """Increase wait time after timeout/failure."""
        self.wait_time = min(self.max_wait, self.wait_time + 1.0)

    def wait(self) -> None:
        """Wait for the current timeout period."""
        time.sleep(self.wait_time)


class MoviePosterGrabber:
    """Class to grab movie poster images from online sources."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        """Initialize the MoviePosterGrabber.

        Args:
            api_key: TMDB API key. If not provided,
                     will try to get from environment variable TMDB_API_KEY
        """
        self.api_key = api_key or os.getenv("TMDB_API_KEY")
        if not self.api_key:
            raise ValueError(
                "TMDB API key must be provided or set in TMDB_API_KEY environment variable"
            )

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
        params = {"api_key": self.api_key, "query": movie_name}
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

        with open(save_path, "wb") as file:
            file.write(response.content)

        return True, False

    def get_movie_details(self, movie_name: str) -> Tuple[Optional[str], Optional[str], bool]:
        """Get the poster URL and description for a given movie name.

        Args:
            movie_name: Name of the movie to search for

        Returns:
            Tuple of (URL of the movie poster if found, movie description if found, whether timeout occurred)
            URL and description will be None if not found or error occurred
        """
        search_url = f"{self.base_url}/search/movie"
        params = {"api_key": self.api_key, "query": movie_name}
        try:
            response = requests.get(search_url, params=params, timeout=5)
        except requests.exceptions.Timeout:
            return None, None, True
        except OSError:
            return None, None, False

        if response.status_code != 200:
            return None, None, False

        results = response.json().get("results", [])
        if not results:
            return None, None, False

        first_result = results[0]
        poster_path = first_result.get("poster_path")
        description = first_result.get("overview")

        poster_url = f"{self.image_base_url}{poster_path}" if poster_path else None
        return poster_url, description, False

    def download_poster_and_description(
        self, movie_name: str, poster_path: str, description_path: str
    ) -> Tuple[bool, bool, bool]:
        """Download the movie poster and description and save them to specified paths.

        Args:
            movie_name: Name of the movie
            poster_path: Path where the poster should be saved
            description_path: Path where the description should be saved

        Returns:
            Tuple of (poster success, description success, whether timeout occurred)
        """
        poster_url, description, timeout = self.get_movie_details(movie_name)
        if timeout:
            return False, False, True

        poster_success = False
        description_success = False

        if poster_url:
            try:
                response = requests.get(poster_url, timeout=5)
                if response.status_code == 200:
                    with open(poster_path, "wb") as file:
                        file.write(response.content)
                    poster_success = True
            except (requests.exceptions.Timeout, OSError):
                pass

        if description:
            try:
                with open(description_path, "w", encoding="utf-8") as file:
                    file.write(description)
                description_success = True
            except OSError:
                pass

        return poster_success, description_success, timeout


if __name__ == "__main__":
    data = pd.read_csv("~/Datasets/MovieLens20M/movie.csv")
    movie_poster_grabber = MoviePosterGrabber()
    ROOT_DIR = "/home/limjk/Datasets/MovieLens20M/posters"
    rate_limiter = RateLimiter()

    # Create descriptions directory if it doesn't exist
    descriptions_dir = os.path.join(ROOT_DIR, "descriptions")
    os.makedirs(descriptions_dir, exist_ok=True)

    for _, row in tqdm(data.iterrows(), total=data.shape[0], desc="Grabbing movie data"):
        movie_id, title = row["movieId"], row["title"]
        # Remove year from title using regex (e.g., "Movie Title (1999)" -> "Movie Title")
        title = re.sub(r"\s*\(\d{4}\)\s*$", "", title)

        forbidden_chars = [" ", "/", ":", "?", "*", "<", ">", "|", "\\", '"', ",", "."]
        striped_movie_name = title
        for char in forbidden_chars:
            striped_movie_name = striped_movie_name.replace(char, "_")

        striped_movie_name = striped_movie_name.replace("__", "_")
        striped_movie_name = striped_movie_name.strip()

        poster_filename = f"{movie_id}_{striped_movie_name}.jpg"
        description_filename = f"{movie_id}_{striped_movie_name}.txt"
        
        poster_path = os.path.join(ROOT_DIR, poster_filename)
        description_path = os.path.join(descriptions_dir, description_filename)

        # Skip if description exists and has content
        if os.path.exists(description_path) and os.path.getsize(description_path) > 0:
            continue

        # Only get description if poster already exists
        if os.path.exists(poster_path) and os.path.getsize(poster_path) > 0:
            _, description, timeout = movie_poster_grabber.get_movie_details(title)
            if timeout:
                rate_limiter.failure()
                rate_limiter.wait()
                continue

            if description:
                try:
                    with open(description_path, "w", encoding="utf-8") as file:
                        file.write(description)
                    rate_limiter.success()
                except OSError:
                    pass
        else:
            poster_success, desc_success, timeout = movie_poster_grabber.download_poster_and_description(
                title, poster_path, description_path
            )

            if poster_success or desc_success:
                rate_limiter.success()
            elif timeout:
                rate_limiter.failure()

        rate_limiter.wait()
