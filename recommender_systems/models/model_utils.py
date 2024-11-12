from typing import Dict

import torch
import torch.nn as nn

import numpy as np

from models.autoencoder import TorchAutoEncoderModel
from models.matrix_factorization import TorchMatrixFactorizationModel
from models.collaborative_filter import CollaborativeFilter
from trainer.dataset_loader import MovieLens20MDatasetLoader


class ModelWrapper:
    """Wrapper class for the model to predict ratings."""

    def __init__(self, model: nn.Module, dataset_path: str) -> None:
        """Initialize the model wrapper.

        Args:
            model: Model instance for the recommendation system
            dataset_path: Path to the dataset
        """
        self.model = model
        self.dataset = MovieLens20MDatasetLoader(dataset_path, subset_ratio=1.0)
        mean_rating = self.dataset.data["rating"].mean()
        self.dataset.data["deviation"] = self.dataset.data["rating"] - mean_rating

        self.rating_matrix = np.zeros(
            (self.dataset.user_ids.shape[0], self.dataset.item_ids.shape[0]),
            dtype=np.float32,
        )
        self.rating_matrix[
            self.dataset.data["userId"].values, self.dataset.data["movieId"].values
        ] = self.dataset.data["deviation"].values

        self.did_inject_user_row = False

    def _predict_matrix_factorization(self, data: Dict[int, float]) -> Dict[int, float]:
        """Predict ratings using the matrix factorization model.

        Args:
            data: Dictionary of movie_id: rating pairs
                {movie_id: rating, ...}

        Returns:
            Dictionary of movie_id: predicted_rating pairs
                {movie_id: predicted_rating, ...}
        """
        self.dataset.inject_user_row(
            data, increase_user_id=not self.did_inject_user_row
        )
        self.did_inject_user_row = True

        device = self.model.W.weight.device
        item_ids = torch.tensor(self.dataset.item_ids, device=device, dtype=torch.long)
        user_ids = torch.tensor(
            self.dataset.user_ids[-1], device=device, dtype=torch.long
        ).repeat(item_ids.shape[0])

        predictions = self.model(user_ids, item_ids).detach().cpu().numpy()

        return {
            int(self.dataset.item_id_reverse_map[item_id]): float(prediction)
            for item_id, prediction in zip(self.dataset.item_ids, predictions)
        }

    def _predict_autoencoder(self, data: Dict[int, float]) -> Dict[int, float]:
        """Predict ratings using the autoencoder model.

        Args:
            data: Dictionary of movie_id: rating pairs
                {movie_id: rating, ...}

        Returns:
            Dictionary of movie_id: predicted_rating pairs
                {movie_id: predicted_rating, ...}
        """
        idx_map = self.dataset.item_id_map
        predictions = self.model.predict(data, idx_map)

        return predictions

    def _predict_collaborative_filter(self, data: Dict[int, float]) -> Dict[int, float]:
        """Predict ratings using the collaborative filter model.

        Args:
            data: Dictionary of movie_id: rating pairs
                {movie_id: rating, ...}

        Returns:
            Dictionary of movie_id: predicted_rating pairs
                {movie_id: predicted_rating, ...}
        """
        if len(self.model.item_ids_map) == 0:
            self.model.set_item_ids_map(self.dataset.item_id_map)

        predictions = self.model.predict(data)

        return predictions

    @torch.no_grad()
    def predict(self, data: Dict[int, float]) -> Dict[int, float]:
        """Predict ratings for the given data.

        Args:
            data: Dictionary of movie_id: rating pairs
                {movie_id: rating, ...}

        Returns:
            Dictionary of movie_id: predicted_rating pairs
                {movie_id: predicted_rating, ...}
        """
        if isinstance(self.model, TorchMatrixFactorizationModel):
            return self._predict_matrix_factorization(data)
        elif isinstance(self.model, TorchAutoEncoderModel):
            return self._predict_autoencoder(data)
        elif isinstance(self.model, CollaborativeFilter):
            return self._predict_collaborative_filter(data)

        return {}


def model_loader(name: str, config: Dict, device: torch.device) -> nn.Module:
    """Load the model from the given configuration.

    Args:
        name: Name of the model to load
        (matrix_factorization, autoencoder, collaborative_filter)
        config: Configuration for the model
        device: Device to load the model

    Returns:
        Model instance
    """
    model_dict = {
        "matrix_factorization": TorchMatrixFactorizationModel,
        "autoencoder": TorchAutoEncoderModel,
        "collaborative_filter": CollaborativeFilter,
    }

    if name not in model_dict:
        raise ValueError(f"Model {name} not found in model_dict")

    if name == "collaborative_filter":
        model = model_dict[name](**dict(config["model"]))
        model.load_weight(config["weight_path"])
        return model

    model_weight = torch.load(
        config["weight_path"], weights_only=True, map_location=device
    )
    model_config = dict(config["model"])

    if name == "matrix_factorization":
        model_config["n_users"] = model_weight["bias_user"].shape[0]
        model_config["n_items"] = model_weight["bias_item"].shape[0]
    elif name == "autoencoder":
        n_hidden, n_items = model_weight["layer1.weight"].shape
        model_config["n_hidden"] = n_hidden
        model_config["n_items"] = n_items

    model = model_dict[name](**model_config).to(device)

    model.load_state_dict(model_weight)

    return model
