"""FastAPI application for Movie recommender system.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""

import base64
from typing import Dict

from fastapi import APIRouter, File, UploadFile

from scripts.model.movie_db import MovieDB


class FastAPIApp:
    UPLOAD_FILE_ROOT = "uploaded_files"

    def __init__(self, movie_db: MovieDB):
        self._router = APIRouter()
        self._router.add_api_route(
            "/get_random_movie", self.get_random_movie, methods=["POST", "OPTIONS"]
        )
        self._router.add_api_route(
            "/submit_rating", self.submit_rating, methods=["POST", "OPTIONS"]
        )
        self.movie_db = movie_db

        self.ratings = {}

    @property
    def router(self):
        return self._router

    async def get_random_movie(self, query_data: Dict) -> Dict:
        result = self.movie_db.get_random_movie()
        if not result:
            return {"error": "No movie found"}

        movie_id, name, img = result

        # Convert bytes to base64 string for JSON serialization
        img_b64 = base64.b64encode(img).decode("utf-8")

        return {
            "id": movie_id,
            "name": name,
            "rating": self.ratings.get(movie_id, None),
            "poster": img_b64,
        }

    async def submit_rating(self, query_data: Dict) -> Dict:
        movie_id = query_data["movie_id"]
        rating = query_data["rating"]

        self.ratings[movie_id] = rating

        return {"success": True, "movie_id": movie_id, "rating": rating}
