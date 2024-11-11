"""FastAPI application for Movie recommender system.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""

import json
import os
import base64

import numpy as np
import torch

from typing import Dict, Optional

from fastapi import APIRouter, File, UploadFile

from model.movie_db import MovieDB
from recommender_systems.trainer.dataset_loader import MovieLens20MDatasetLoader
from recommender_systems.trainer.train_pytorch_matrix_factorization import (
    TorchMatrixFactorizationModel,
)
from recommender_systems.trainer.train_pytorch_autoencoder import (
    TorchAutoEncoderModel,
)


class FastAPIApp:
    UPLOAD_FILE_ROOT = "uploaded_files"

    def __init__(self, movie_db: MovieDB, rating_path: Optional[str]) -> None:
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
        self.rating_path = rating_path

        if rating_path and os.path.exists(rating_path):
            with open(rating_path, "r") as f:
                self.ratings = json.load(f)

            self.ratings = {int(k): v for k, v in self.ratings.items()}
        else:
            self.ratings = {}

        dataset_path = "~/Datasets/MovieLens20M/rating.csv"
        # self.model_path = "../res/models/matrix_factorization_model.pth"
        self.model_path = "../res/models/autoencoder_model.pth"
        self.dataset = MovieLens20MDatasetLoader(dataset_path, subset_ratio=1.0)

        # if self.ratings:
        #     self.dataset.inject_user_row(self.ratings, increase_user_id=True)

        global_item_bias = np.mean(self.dataset.data["rating"].values)  # type: ignore
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        # self.model = TorchMatrixFactorizationModel(
        #     self.dataset.user_ids.shape[0] + (0 if self.ratings else 1),
        #     self.dataset.item_ids.shape[0],
        #     10,
        #     global_item_bias,
        # ).to(device)
        self.model = TorchAutoEncoderModel(
            self.dataset.item_ids.shape[0],
            512,
            global_item_bias,
        ).to(device)

        if os.path.exists(self.model_path):
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=device, weights_only=True)
            )
            print(f"Model loaded from {self.model_path}")

        self.predictions = self.model.predict(self.ratings, idx_map=self.dataset.item_id_map)
        # self.predictions = self.model.train_and_predict(
        #     self.dataset, epochs=10, lr=100.0, save_path=self.model_path
        # )
        self.sorted_predictions = sorted(
            self.predictions.items(), key=lambda x: x[1], reverse=True
        )

    @property
    def router(self) -> APIRouter:
        """Return the APIRouter instance."""
        return self._router

    async def get_random_movie(self, query_data: Dict) -> Dict:
        best_movie_ids = [
            (int(movie_id), rating)
            for movie_id, rating in self.sorted_predictions[:1000]
        ]
        idx = np.random.choice(len(best_movie_ids))
        movie_id, predicted_rating = best_movie_ids[idx]

        print(f"Chosen idx: {idx}, Movie ID: {movie_id}, Predicted rating: {predicted_rating}")

        predicted_rating = float(
            np.round(np.clip(predicted_rating, 0.5, 5.0) * 2) / 2.0
        )

        result = self.movie_db.get_movie_poster(movie_id)
        # movie_id = -1
        # result = None

        if not result:
            result = self.movie_db.get_random_movie()
            if not result:
                return {"error": "No movie found"}
            print(f"Couldn't find movie with ID: {movie_id}. Using random movie.")
            movie_id, name, img = result
            predicted_rating = None
        else:
            name, img = result

        print(f"Sending movie: {name} (ID: {movie_id}), {type(movie_id)=}")

        # Convert bytes to base64 string for JSON serialization
        img_b64 = base64.b64encode(img).decode("utf-8")

        return {
            "id": movie_id,
            "name": name,
            "rating": self.ratings.get(movie_id, None),
            "predicted_rating": predicted_rating,
            "poster": img_b64,
        }

    async def get_movie(self, query_data: Dict) -> Dict:
        movie_id = int(query_data["movie_id"])
        result = self.movie_db.get_movie_poster(movie_id)

        if not result:
            return {"error": "No movie found"}
        name, img = result

        # Convert bytes to base64 string for JSON serialization
        img_b64 = base64.b64encode(img).decode("utf-8")
        predicted_rating = self.predictions.get(movie_id, None)
        if predicted_rating is not None:
            predicted_rating = float(
                np.round(np.clip(predicted_rating, 0.5, 5.0) * 2) / 2.0
            )

        return {
            "id": movie_id,
            "name": name,
            "rating": self.ratings.get(movie_id, None),
            "predicted_rating": predicted_rating,
            "poster": img_b64,
        }

    async def get_my_ratings(self, query_data: Dict) -> Dict:
        predictions = {
            movie_id: self.predictions.get(movie_id, None)
            for movie_id in self.ratings.keys()
        }
        for movie_id in predictions.keys():
            if predictions[movie_id] is not None:
                predictions[movie_id] = float(
                    np.round(np.clip(predictions[movie_id], 0.5, 5.0) * 2) / 2.0
                )

        return {
            "ratings": self.ratings,
            "predicted_ratings": predictions,
        }

    async def submit_rating(self, query_data: Dict) -> Dict:
        movie_id = int(query_data["movie_id"])
        rating = float(query_data["rating"])

        increase_user_id = False
        if not self.ratings:
            increase_user_id = True

        self.ratings[movie_id] = rating
        self.dataset.inject_user_row(self.ratings, increase_user_id=increase_user_id)

        if self.rating_path:
            with open(self.rating_path, "w") as f:
                f.write(json.dumps(self.ratings))

        self.predictions = self.model.predict(self.ratings, idx_map=self.dataset.item_id_map)
        # self.predictions = self.model.train_and_predict(
        #     self.dataset, epochs=10, lr=100.0, save_path=self.model_path
        # )
        self.sorted_predictions = sorted(
            self.predictions.items(), key=lambda x: x[1], reverse=True
        )

        predicted_rating = self.predictions.get(movie_id, None)
        if predicted_rating is not None:
            predicted_rating = float(
                np.round(np.clip(predicted_rating, 0.5, 5.0) * 2.0) / 2.0
            )

        return {
            "success": True,
            "movie_id": movie_id,
            "rating": rating,
            "predicted_rating": predicted_rating,
        }
