"""Collaborative filtering model for movie recommendation system.

Author: Jongkuk Lim
Contact: lim.jeikei@gmail.com
"""
import json
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from trainer.dataset_loader import MovieLens20MDatasetLoader


class CollaborativeFilter:
    """Collaborative filtering model for movie recommendation system."""

    MIN_OVERLAP = 5
    """Minimum number of common users between two items to calculate weights."""

    def __init__(
        self,
        mean_rating: float,
        top_k: int,
    ) -> None:
        """Initialize the collaborative filtering model.

        Args:
            mean_rating: Mean rating of the dataset
            top_k: Number of similar items to consider for prediction
        """
        self.mean_rating = mean_rating
        self.top_k = top_k
        self.weights: Dict[Tuple[int, int], float] = {}
        self.item_ids_map: Dict[int, int] = {}
        self.item_ids_reverse_map: Dict[int, int] = {}

    def set_item_ids_map(self, item_ids_map: Dict[int, int]) -> None:
        """Set the item_ids map for the model.

        This must be called before calculating weights and predicting.

        Args:
            item_ids_map: Mapping of item_id to index in the model
                {item_id: index, ...}
        """
        self.item_ids_map = item_ids_map
        self.item_ids_reverse_map = {v: k for k, v in item_ids_map.items()}

    def load_weight(self, weight_path: str) -> None:
        """Load the weights from the given path."""
        if os.path.exists(weight_path):
            with open(weight_path, "rb") as f:
                self.weights = pickle.load(f)

    def save_weight(self, weight_path: str) -> None:
        """Save the weights to the given path."""
        with open(weight_path, "wb") as f:
            pickle.dump(self.weights, f)

    def calculate_weights(self, item_ids: List[int], rating_matrix: np.ndarray) -> None:
        """Calculate the weights between the given items.

        Args:
            item_ids: List of item_ids to calculate weights
            rating_matrix: Rating matrix of the dataset (n_users, n_items)
        """
        rating_mask = rating_matrix != 0

        item_ids = [self.item_ids_map[i] for i in item_ids]

        p_bar_i = tqdm(item_ids, desc="Calculating weights for item i")
        for i in p_bar_i:
            p_bar_i.set_postfix({"item_id": i})
            for j in item_ids:
                if i == j:
                    continue
                if (i, j) in self.weights:
                    continue

                common_user_id = np.where(rating_mask[:, [i, j]].all(axis=1))[0]

                if len(common_user_id) < CollaborativeFilter.MIN_OVERLAP:
                    continue

                common_item_data_i = rating_matrix[common_user_id, i]
                common_item_data_j = rating_matrix[common_user_id, j]

                numerator = np.dot(common_item_data_i, common_item_data_j)
                denominator = np.linalg.norm(common_item_data_i) * np.linalg.norm(
                    common_item_data_j
                )

                weight = numerator / (denominator + 1e-9)
                self.weights[(i, j)] = weight
                self.weights[(j, i)] = weight

    def predict(self, ratings: Dict[int, float]) -> Dict[int, float]:
        """Predict the ratings for the given items.

        Args:
            ratings: Dictionary of item_id: rating pairs
                {item_id: rating, ...}

        Returns:
            {item_id: predicted_rating, ...}
        """
        predictions = {}

        for item_id in ratings.keys():
            sorted_weights = []
            for cmp_id in ratings.keys():
                if cmp_id == item_id:
                    continue

                item_key = (self.item_ids_map[item_id], self.item_ids_map[cmp_id])

                if item_key not in self.weights.keys():
                    continue

                if (
                    item_key[0],
                    self.weights[(item_key[1], item_key[0])],
                ) in sorted_weights:
                    continue

                sorted_weights.append((cmp_id, self.weights[item_key]))
                sorted_weights.sort(key=lambda x: x[1], reverse=True)

                if len(sorted_weights) > self.top_k:
                    sorted_weights = sorted_weights[: self.top_k]

            numerator = 0.0
            sum_weights = sum(abs(weight) for _, weight in sorted_weights)
            for cmp_id, weight in sorted_weights:
                numerator += weight * (ratings[cmp_id] - self.mean_rating)

            if sum_weights == 0:
                prediction = self.mean_rating
            else:
                prediction = self.mean_rating + numerator / sum_weights

            predictions[item_id] = prediction

        return predictions


if __name__ == "__main__":
    PATH = "~/Datasets/MovieLens20M/rating.csv"

    dataset = MovieLens20MDatasetLoader(PATH, subset_ratio=1.0)

    with open("../res/ratings.json", "r", encoding="utf-8") as main_fp:
        user_ratings = json.load(main_fp)

    user_ratings = {int(k): v for k, v in user_ratings.items()}

    main_mean_rating = dataset.data["rating"].mean()

    dataset.data["deviation"] = dataset.data["rating"] - main_mean_rating

    main_rating_matrix: np.ndarray = np.zeros(
        (dataset.user_ids.shape[0], dataset.item_ids.shape[0]), dtype=np.float32
    )
    main_rating_matrix[
        dataset.data["userId"].values, dataset.data["movieId"].values
    ] = dataset.data["deviation"].values

    model = CollaborativeFilter(float(main_mean_rating), 30)
    model.set_item_ids_map(dataset.item_id_map)
    model.load_weight("../res/models/cf_weights.pkl")
    model.calculate_weights(list(user_ratings.keys()), main_rating_matrix)
    model.save_weight("../res/models/cf_weights.pkl")

    main_predictions = model.predict(user_ratings)

    mse = sum((main_predictions[k] - v) ** 2 for k, v in user_ratings.items()) / len(
        user_ratings
    )
    print(f"MSE: {mse}, RMSE: {np.sqrt(mse)}")
