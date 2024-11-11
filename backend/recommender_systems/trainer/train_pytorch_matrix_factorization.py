import os
import json

import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from typing import Dict

from recommender_systems.trainer.dataset_loader import MovieLens20MDatasetLoader


class TorchMatrixFactorizationModel(nn.Module):
    """Matrix Factorization model using PyTorch."""

    def __init__(self, n_users: int, n_items: int, k: int, global_mean: float) -> None:
        """Initialize the model.

        Args:
            n_users: Number of users
            n_items: Number of items
            k: Number of latent dimensions
            global_mean: Global mean of the ratings
        """
        super().__init__()

        self.W = nn.Embedding(n_users, k)
        self.U = nn.Embedding(n_items, k)

        self.bias_user = nn.Parameter(torch.zeros(n_users, dtype=torch.float32))
        self.bias_item = nn.Parameter(
            torch.zeros(n_items, requires_grad=True, dtype=torch.float32)
        )
        self.global_mean = nn.Parameter(
            torch.tensor(global_mean, dtype=torch.float32), requires_grad=False
        )

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        wu = torch.einsum("ij, ij -> i", self.W(user_ids), self.U(item_ids))
        return (
            self.bias_user[user_ids] + self.bias_item[item_ids] + self.global_mean + wu
        )

    def train_and_predict(
        self,
        dataset: MovieLens20MDatasetLoader,
        # ratings: Dict[int, float],
        epochs: int = 500,
        lr: float = 100.0,
        save_path: str = "model.pth",
    ) -> Dict[int, float]:
        """Train the model on the given ratings and predict the missing ratings.

        Args:
            dataset_path: Path to the dataset file
            ratings: Dictionary containing the ratings for the user
        """
        # dataset.inject_user_row(ratings)
        _, train_set = dataset.get_train_test_split(test_size=1.0, shuffle_set=True)
        device = self.W.weight.device

        train_user_ids = torch.tensor(train_set.data["userId"].values, device=device, dtype=torch.long)
        train_item_ids = torch.tensor(train_set.data["movieId"].values, device=device, dtype=torch.long)
        train_ratings = torch.tensor(
            train_set.data["rating"].values, device=device, dtype=torch.float32
        )

        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        p_bar = tqdm(range(epochs), desc="Training")
        for _ in p_bar:
            prediction = self(train_user_ids, train_item_ids)
            # prediction = torch.clamp(prediction, 0.5, 5.0)

            train_loss = loss_fn(prediction, train_ratings)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            p_bar.set_postfix({"Loss(Train)": train_loss.item()})

        torch.save(self.state_dict(), save_path)

        with torch.no_grad():
            item_ids = torch.tensor(dataset.item_ids, device=device)
            user_ids = torch.tensor(dataset.user_ids[-1], device=device).repeat(
                item_ids.shape[0]
            )

            predictions = self(user_ids, item_ids).cpu().numpy()

        # predictions = np.clip(predictions, 0.5, 5.0)
        # Make predictions by 0.5 steps
        # predictions = np.round(predictions * 2) / 2

        return {
            dataset.item_id_reverse_map[idx]: rating
            for idx, rating in zip(dataset.item_ids, predictions)
        }


if __name__ == "__main__":
    path = "~/Datasets/MovieLens20M/rating.csv"
    model_path = "../res/models/matrix_factorization_model.pth"

    dataset = MovieLens20MDatasetLoader(path, subset_ratio=1.0)

    with open("../res/ratings.json", "r") as f:
        user_ratings = json.load(f)

    user_ratings = {int(k): v for k, v in user_ratings.items()}

    K = 10
    EPOCHS = 500
    LR = 100.0

    global_item_bias = np.mean(dataset.data["rating"].values)  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TorchMatrixFactorizationModel(
        dataset.user_ids.shape[0] + 1, dataset.item_ids.shape[0], K, global_item_bias
    ).to(device)

    if os.path.exists(model_path):
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        print("Model loaded from model.pth")

    predictions = model.train_and_predict(
        dataset, user_ratings, epochs=EPOCHS, lr=LR, save_path=model_path
    )
    rmse = np.sqrt(
        np.mean(
            [
                (rating - predictions[item_id]) ** 2
                for item_id, rating in user_ratings.items()
            ]
        )
    )

    print(f"RMSE: {rmse}")
