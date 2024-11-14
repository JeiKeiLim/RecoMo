"""Train a PyTorch AutoEncoder model on the MovieLens 20M dataset.

Author: Jongkuk Lim
Contact: lim.jeikei@gmail.com
"""
import json

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from models.autoencoder import TorchAutoEncoderModel
from trainer.dataset_loader import (MovieLens20MDataset,
                                    MovieLens20MDatasetLoader)

if __name__ == "__main__":
    PATH = "~/Datasets/MovieLens20M/rating.csv"
    MODEL_PATH = "../res/models/autoencoder_model.pth"

    dataset = MovieLens20MDatasetLoader(PATH, subset_ratio=1.0)

    with open("../res/ratings.json", "r", encoding="utf-8") as f:
        user_ratings = json.load(f)

    user_ratings = {int(k): v for k, v in user_ratings.items()}
    dataset.inject_user_row(user_ratings, increase_user_id=True)

    unique_test_user_ids = dataset.data["userId"].max()
    unique_test_user_ids = np.array(unique_test_user_ids)[np.newaxis]

    EPOCHS = 100
    LR = 0.1
    BATCH_SIZE = 10240
    N_HIDDEN = 512

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    train_set = MovieLens20MDataset(
        dataset.data.query(f"userId < {unique_test_user_ids[0]}")
    )
    test_set = MovieLens20MDataset(
        dataset.data.query(f"userId >= {unique_test_user_ids[0]}")
    )

    # WARNING: This takes about 13.8*2 GB of memory
    rating_matrix: np.ndarray = np.zeros(
        (dataset.user_ids.shape[0], dataset.item_ids.shape[0]), dtype=np.float32
    )
    rating_matrix[
        dataset.data["userId"].values, dataset.data["movieId"].values
    ] = dataset.data["rating"].values
    rating_matrix_tensor = torch.tensor(
        rating_matrix, dtype=torch.float32, device="cpu"
    )
    rating_mask = rating_matrix_tensor != 0

    unique_train_user_ids = train_set.data["userId"].unique()

    mean_train_rating = torch.tensor(
        train_set.data["rating"].mean(), device=device, dtype=torch.float32
    )

    model = TorchAutoEncoderModel(  # pylint: disable=invalid-name
        dataset.item_ids.shape[0], N_HIDDEN, mean_train_rating.item()
    ).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    loss_fn = nn.MSELoss()

    p_bar = tqdm(range(EPOCHS), desc="Training")
    for _ in p_bar:
        train_losses = []
        for i in range(0, unique_train_user_ids.shape[0], BATCH_SIZE):
            i_batch = i + BATCH_SIZE
            batch_user_ids = unique_train_user_ids[i:i_batch]

            mat = rating_matrix_tensor[batch_user_ids].to(device)  # type: ignore
            mask = rating_mask[batch_user_ids].to(device)  # type: ignore
            mat[mask] -= mean_train_rating

            prediction = model(mat, is_train=True)  # pylint: disable=not-callable

            loss = loss_fn(prediction[mask], mat[mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append([loss.item()])

        test_losses = []
        with torch.no_grad():
            for i in range(0, unique_test_user_ids.shape[0], BATCH_SIZE):
                i_batch = i + BATCH_SIZE
                batch_user_ids = unique_test_user_ids[i:i_batch]

                mat = rating_matrix[batch_user_ids].to(device)  # type: ignore
                mask = rating_mask[batch_user_ids].to(device)  # type: ignore
                mat[mask] -= mean_train_rating

                prediction = model(mat, is_train=False)  # pylint: disable=not-callable

                loss = loss_fn(prediction[mask], mat[mask])

                test_losses.append([loss.item()])

        p_bar.set_postfix(
            {"Loss(Train)": np.mean(train_losses), "Loss(Test)": np.mean(test_losses)}
        )

    test_losses = []
    with torch.no_grad():
        for i in range(0, unique_test_user_ids.shape[0], BATCH_SIZE):
            i_batch = i + BATCH_SIZE
            batch_user_ids = unique_test_user_ids[i:i_batch]

            mat = rating_matrix[batch_user_ids].to(device)  # type: ignore
            mask = rating_mask[batch_user_ids].to(device)  # type: ignore
            mat[mask] -= mean_train_rating

            prediction = model(mat, is_train=False)  # pylint: disable=not-callable

            loss = loss_fn(prediction[mask], mat[mask])

            test_losses.append([loss.item()])

    torch.save(model.state_dict(), MODEL_PATH)

    print(
        f"Final Test Loss: {np.mean(test_losses)}, RMSE: {np.sqrt(np.mean(test_losses))}"
    )
