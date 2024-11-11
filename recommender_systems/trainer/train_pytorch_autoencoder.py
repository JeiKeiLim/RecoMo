import json

import numpy as np
import torch
import torch.nn as nn
from models.autoencoder import TorchAutoEncoderModel
from tqdm import tqdm
from trainer.dataset_loader import MovieLens20MDataset, MovieLens20MDatasetLoader

if __name__ == "__main__":
    path = "~/Datasets/MovieLens20M/rating.csv"
    model_path = "../res/models/matrix_factorization_model.pth"

    dataset = MovieLens20MDatasetLoader(path, subset_ratio=1.0)

    with open("../res/ratings.json", "r") as f:
        user_ratings = json.load(f)

    user_ratings = {int(k): v for k, v in user_ratings.items()}
    dataset.inject_user_row(user_ratings, increase_user_id=True)

    unique_test_user_ids = dataset.data["userId"].max()
    unique_test_user_ids = np.array(unique_test_user_ids)[np.newaxis]

    EPOCHS = 100
    LR = 0.1
    batch_size = 10240

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    train_set = MovieLens20MDataset(
        dataset.data.query(f"userId < {unique_test_user_ids[0]}")
    )
    test_set = MovieLens20MDataset(
        dataset.data.query(f"userId >= {unique_test_user_ids[0]}")
    )

    # WARNING: This takes about 13.8*2 GB of memory
    rating_matrix = np.zeros(
        (dataset.user_ids.shape[0], dataset.item_ids.shape[0]), dtype=np.float32
    )
    rating_matrix[
        dataset.data["userId"].values, dataset.data["movieId"].values
    ] = dataset.data["rating"].values
    rating_matrix = torch.tensor(rating_matrix, dtype=torch.float32, device="cpu")
    rating_mask = rating_matrix != 0

    unique_train_user_ids = train_set.data["userId"].unique()

    mean_train_rating = torch.tensor(
        train_set.data["rating"].mean(), device=device, dtype=torch.float32
    )

    model = TorchAutoEncoderModel(
        dataset.item_ids.shape[0], 512, mean_train_rating.item()
    ).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    loss_fn = nn.MSELoss()

    p_bar = tqdm(range(EPOCHS), desc="Training")
    for _ in p_bar:
        train_losses = []
        for i in range(0, unique_train_user_ids.shape[0], batch_size):
            mat = rating_matrix[unique_train_user_ids[i : i + batch_size]].to(device)  # type: ignore
            mask = rating_mask[unique_train_user_ids[i : i + batch_size]].to(device)  # type: ignore
            mat[mask] -= mean_train_rating

            prediction = model(mat, is_train=True)

            loss = loss_fn(prediction[mask], mat[mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append([loss.item()])

        test_losses = []
        with torch.no_grad():
            for i in range(0, unique_test_user_ids.shape[0], batch_size):
                mat = rating_matrix[unique_test_user_ids[i : i + batch_size]].to(device)  # type: ignore
                mask = rating_mask[unique_test_user_ids[i : i + batch_size]].to(device)  # type: ignore
                mat[mask] -= mean_train_rating

                prediction = model(mat, is_train=False)

                loss = loss_fn(prediction[mask], mat[mask])

                test_losses.append([loss.item()])

        p_bar.set_postfix(
            {"Loss(Train)": np.mean(train_losses), "Loss(Test)": np.mean(test_losses)}
        )

    test_losses = []
    with torch.no_grad():
        for i in range(0, unique_test_user_ids.shape[0], batch_size):
            mat = rating_matrix[unique_test_user_ids[i : i + batch_size]].to(device)  # type: ignore
            mask = rating_mask[unique_test_user_ids[i : i + batch_size]].to(device)  # type: ignore
            mat[mask] -= mean_train_rating

            prediction = model(mat, is_train=False)

            loss = loss_fn(prediction[mask], mat[mask])

            test_losses.append([loss.item()])

    torch.save(model.state_dict(), "../res/models/autoencoder_model.pth")

    print(
        f"Final Test Loss: {np.mean(test_losses)}, RMSE: {np.sqrt(np.mean(test_losses))}"
    )
