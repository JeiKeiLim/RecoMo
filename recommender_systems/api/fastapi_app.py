"""FastAPI application for Movie recommender system.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""

import json
import os
from typing import Dict, Optional

import torch.nn as nn
from fastapi import APIRouter, File, UploadFile
from models.model_utils import ModelWrapper
from trainer.dataset_loader import MovieLens20MDatasetLoader


class FastAPIApp:
    UPLOAD_FILE_ROOT = "uploaded_files"

    def __init__(self, rating_path: Optional[str], model: nn.Module) -> None:
        self._router = APIRouter()
        self._router.add_api_route(
            "/post_rating", self.post_rating, methods=["POST", "OPTIONS"]
        )
        self._router.add_api_route(
            "/get_prediction", self.get_prediction, methods=["POST", "OPTIONS"]
        )
        self.rating_path = rating_path

        if rating_path and os.path.exists(rating_path):
            with open(rating_path, "r") as f:
                self.ratings = json.load(f)

            self.ratings = {int(k): v for k, v in self.ratings.items()}
        else:
            self.ratings = {}

        dataset_path = "~/Datasets/MovieLens20M/rating.csv"
        # self.dataset = MovieLens20MDatasetLoader(dataset_path, subset_ratio=1.0)

        self.model = ModelWrapper(model, dataset_path)
        self.predictions = self.model.predict(self.ratings)
        self.sorted_predictions = sorted(
            self.predictions.items(), key=lambda x: x[1], reverse=True
        )

    @property
    def router(self) -> APIRouter:
        """Return the APIRouter instance."""
        return self._router

    def post_rating(self, query_data: Dict) -> Dict:
        """Post the rating of the movie."""
        movie_id = query_data.get("movie_id", None)
        rating = query_data.get("rating", None)

        if movie_id is None or rating is None:
            return {"status": "failed", "message": "movie_id or rating is missing."}

        self.ratings[movie_id] = rating

        if self.rating_path:
            with open(self.rating_path, "w") as f:
                json.dump(self.ratings, f)

        self.predictions = self.model.predict(self.ratings)
        self.sorted_predictions = sorted(
            self.predictions.items(), key=lambda x: x[1], reverse=True
        )

        return {"status": "success"}

    def get_prediction(self, query_data: Dict) -> Dict:
        """Get the prediction of the movie."""
        movie_id = query_data["movie_id"]

        if movie_id >= 0:
            return {
                "success": True,
                "predictions": {movie_id: self.predictions.get(movie_id, None)},
            }

        return {"success": True, "predictions": self.predictions}
