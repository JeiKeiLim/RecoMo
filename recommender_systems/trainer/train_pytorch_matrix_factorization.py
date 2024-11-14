"""Train a matrix factorization model using PyTorch.

Author: Jongkuk Lim
Contact: lim.jeikei@gmail.com
"""
import json
import os

import numpy as np
import torch

from models.matrix_factorization import TorchMatrixFactorizationModel
from trainer.dataset_loader import MovieLens20MDatasetLoader

if __name__ == "__main__":
    PATH = "~/Datasets/MovieLens20M/rating.csv"
    MODEL_PATH = "../res/models/matrix_factorization_model.pth"

    dataset = MovieLens20MDatasetLoader(PATH, subset_ratio=1.0)

    with open("../res/ratings.json", "r", encoding="utf-8") as f:
        user_ratings = json.load(f)

    user_ratings = {int(k): v for k, v in user_ratings.items()}
    dataset.inject_user_row(ratings=user_ratings, increase_user_id=True)

    K = 10
    EPOCHS = 500
    LR = 100.0

    global_item_bias = np.mean(dataset.data["rating"].values)  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TorchMatrixFactorizationModel(  # pylint: disable=invalid-name
        dataset.user_ids.shape[0], dataset.item_ids.shape[0], K, float(global_item_bias)
    ).to(device)

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location=device, weights_only=True)
        )
        print("Model loaded from model.pth")

    predictions = model.train_anlearning_ratepredict(
        dataset, epochs=EPOCHS, lr=LR, save_path=MODEL_PATH
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
