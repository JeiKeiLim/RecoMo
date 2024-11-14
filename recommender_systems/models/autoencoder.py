"""Autoencoder model for movie recommendation system.

Author: Jongkuk Lim
Contact: lim.jeikei@gmail.com
"""
from typing import Dict

import torch
from torch import nn


class TorchAutoEncoderModel(nn.Module):
    """Autoencoder model for movie recommendation system."""

    def __init__(self, n_items: int, n_hidden: int, global_mean: float) -> None:
        """Initialize the Autoencoder model.

        Args:
            n_items: Number of items in the dataset
            n_hidden: Number of hidden units in the model
            global_mean: Global mean of the ratings
        """
        super().__init__()
        self.n_items = n_items
        self.layer1 = nn.Linear(n_items, n_hidden)
        self.activation = nn.Tanh()
        self.layer2 = nn.Linear(n_hidden, n_items)
        self.dropout = nn.Dropout(0.7)
        self.global_mean = global_mean

    def forward(self, x: torch.Tensor, is_train: bool = False) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, n_items)
            is_train: Whether the model is in training mode
                      If true, applies dropout

        Returns:
            Output tensor of shape (batch_size, n_items)
        """
        if is_train:
            out = self.dropout(x)
        else:
            out = x

        out = self.layer1(out)
        out = self.activation(out)
        out = self.layer2(out)

        return out

    @torch.no_grad()
    def predict(
        self, data: Dict[int, float], idx_map: Dict[int, int]
    ) -> Dict[int, float]:
        """Predict ratings for the given data.

        Args:
            data: Dictionary of movie_id: rating pairs
                {movie_id: rating, ...}
            idx_map: Mapping of movie_id to index in the model

        Returns:
            Dictionary of movie_id: predicted_rating pairs
            {movie_id: predicted_rating, ...
        """
        mat = torch.zeros((1, self.n_items), dtype=torch.float32).to(
            self.layer1.weight.device
        )
        for item_id, rating in data.items():
            mat[0, idx_map[item_id]] = rating - self.global_mean

        prediction = self(mat, is_train=False) + self.global_mean
        reverse_idx_map = {v: k for k, v in idx_map.items()}
        return {
            int(reverse_idx_map[i]): float(prediction[0, i].item())
            for i in range(self.n_items)
        }
