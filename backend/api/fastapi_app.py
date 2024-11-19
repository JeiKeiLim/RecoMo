"""FastAPI application for Movie recommender system.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""

import base64
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
import urllib3
from fastapi import APIRouter

from model.movie_db import MovieDB


class FastAPIApp:
    """FastAPI application for Movie recommender system backend."""

    UPLOAD_FILE_ROOT = "uploaded_files"
    ENGINE_API_URL = "http://localhost:8888"

    def __init__(self, movie_db: MovieDB, rating_path: Optional[str]) -> None:
        """Initialize FastAPI application.

        Args:
            movie_db: Movie database instance.
            rating_path: Path to the rating file.
                Rating json file should have the following format:
                {
                    "movie_id": rating,
                    ...
                }
        """
        self._router = APIRouter()
        self._router.add_api_route(
            "/get_random_movie", self.get_random_movie, methods=["POST", "OPTIONS"]
        )
        self._router.add_api_route(
            "/submit_rating", self.submit_rating, methods=["POST", "OPTIONS"]
        )
        self._router.add_api_route(
            "/get_my_ratings", self.get_my_ratings, methods=["POST", "OPTIONS"]
        )
        self._router.add_api_route(
            "/get_movie", self.get_movie, methods=["POST", "OPTIONS"]
        )
        self.movie_db = movie_db

        if rating_path and os.path.exists(rating_path):
            with open(rating_path, "r", encoding="utf-8") as f:
                self.ratings = json.load(f)

            self.ratings = {int(k): v for k, v in self.ratings.items()}
        else:
            self.ratings = {}

        self.predictions: Dict[int, float] = {}
        """Prediction results from the engine API.
        {
            movie_id(int): rating(float),
        }
        """
        self.sorted_predictions: List[Tuple[int, float]] = []
        """Sorted prediction results.

        List[Tuple[movie_id(int), rating(float)]]
        """

        self._update_predictions()

    def _sort_predictions(self) -> None:
        """Sort predictions by rating."""
        self.sorted_predictions = sorted(
            self.predictions.items(), key=lambda x: x[1], reverse=True
        )

    def _clip_rating(self, rating: float) -> float:
        """Clip rating to 0.5 ~ 5.0."""
        return float(np.round(np.clip(rating, 0.5, 5.0) * 2) / 2.0)

    def _update_predictions(self) -> bool:
        """Update predictions from the engine API."""
        prediction_result = self.post_engine_api("/get_prediction", {"movie_id": -1})

        if prediction_result.get("success", False):
            predictions = prediction_result["predictions"]
            self.predictions = {
                int(movie_id): rating for movie_id, rating in predictions.items()
            }
            self._sort_predictions()

            return True

        return False

    @property
    def router(self) -> APIRouter:
        """Return the APIRouter instance."""
        return self._router

    def post_engine_api(self, endpoint: str, data: Dict) -> Dict:
        """Post data to the engine API.

        Args:
            endpoint: Endpoint to post.
            data: Data to post.
        """
        try:
            response = requests.post(  # pylint: disable=missing-timeout
                f"{self.ENGINE_API_URL}/{endpoint}",
                json=data,
            )
        except urllib3.exceptions.MaxRetryError:
            return {}

        return response.json()

    async def get_random_movie(self, _: Dict) -> Dict:
        """Get random movie.

        Args:
            query_data: Not used for now.

        Returns:
            {
                "id": movie_id(int),
                "name": movie_name(str),
                "rating": rating(float),
                "predicted_rating": predicted_rating(float),
                "poster": poster_image (base64(str)),
            }
        """
        predicted_rating: Optional[float]

        best_movie_ids = [
            (int(movie_id), rating)
            for movie_id, rating in self.sorted_predictions[:1000]
        ]
        idx = np.random.choice(len(best_movie_ids))
        movie_id, predicted_rating = best_movie_ids[idx]

        print(
            f"Chosen idx: {idx}, Movie ID: {movie_id}, Predicted rating: {predicted_rating}"
        )

        predicted_rating = self._clip_rating(predicted_rating)
        result = self.movie_db.get_movie_poster(movie_id)
        description = ""

        if not result:
            result_random = self.movie_db.get_random_movie()

            if not result_random:
                return {"error": "No movie found"}

            movie_id, name, img = result_random
            predicted_rating = None
        else:
            name, img = result

        description_result = self.movie_db.get_movie_description(movie_id)

        if description_result:
            description = description_result[1]

        print(f"Sending movie: {name} (ID: {movie_id}), {type(movie_id)=}, {description=}")

        # Convert bytes to base64 string for JSON serialization
        img_b64 = base64.b64encode(img).decode("utf-8")

        return {
            "id": movie_id,
            "name": name,
            "rating": self.ratings.get(movie_id, None),
            "predicted_rating": predicted_rating,
            "poster": img_b64,
            "description": description,
        }

    async def get_movie(self, query_data: Dict) -> Dict:
        """Get movie by ID.

        Args:
            query_data: {
                "movie_id": movie_id(int),
                }

        Returns:
            {
                "id": movie_id(int),
                "name": movie_name(str),
                "rating": rating(float),
                "predicted_rating": predicted_rating(float),
                "poster": poster_image (base64(str)),
            }
        """
        movie_id = int(query_data["movie_id"])
        result = self.movie_db.get_movie_poster(movie_id)
        description = ""

        if not result:
            return {"error": "No movie found"}
        name, img = result

        description_result = self.movie_db.get_movie_description(movie_id)
        if description_result:
            description = description_result[1]

        # Convert bytes to base64 string for JSON serialization
        img_b64 = base64.b64encode(img).decode("utf-8")

        predicted_rating = self.predictions.get(movie_id, None)
        if predicted_rating is not None:
            predicted_rating = self._clip_rating(predicted_rating)

        return {
            "id": movie_id,
            "name": name,
            "rating": self.ratings.get(movie_id, None),
            "predicted_rating": predicted_rating,
            "poster": img_b64,
            "description": description,
        }

    async def get_my_ratings(self, _: Dict) -> Dict:
        """Get list of my ratings.

        Args:
            query_data: Not used for now.

        Returns:
            {
                "ratings": {
                    movie_id(int): rating(float),
                },
                "predicted_ratings": {
                    movie_id(int): predicted_rating(float),
                },
            }
        """
        predictions = {
            movie_id: self.predictions.get(movie_id, None)
            for movie_id in self.ratings.keys()
        }

        for movie_id in predictions.keys():
            if predictions[movie_id] is not None:
                predictions[movie_id] = self._clip_rating(predictions[movie_id])  # type: ignore

        return {
            "ratings": self.ratings,
            "predicted_ratings": predictions,
        }

    async def submit_rating(self, query_data: Dict) -> Dict:
        """Submit rating from frontend.

        Args:
            query_data: {
                "movie_id": movie_id(int),
                "rating": rating(float),
                }

        Returns:
            {
                "success": (bool),
                "movie_id": movie_id(int),
                "rating": rating(float),
                "predicted_rating": predicted_rating(float),
            }
        """
        movie_id = int(query_data["movie_id"])
        rating = float(query_data["rating"])

        self.ratings[movie_id] = rating

        result = self.post_engine_api(
            "/post_rating", {"movie_id": movie_id, "rating": rating}
        )

        if result.get("success", False):
            self._update_predictions()
        else:
            return {"success": False}

        predicted_rating = self.predictions.get(movie_id, None)
        if predicted_rating is not None:
            predicted_rating = self._clip_rating(predicted_rating)

        return {
            "success": True,
            "movie_id": movie_id,
            "rating": rating,
            "predicted_rating": predicted_rating,
        }
