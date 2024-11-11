import pandas as pd
import numpy as np

from typing import Tuple, Union, List, Dict


class MovieLens20MDatasetLoader:
    """Class to load the MovieLens 20M dataset."""

    def __init__(self, path: str, subset_ratio: float = 1.0) -> None:
        """Load the MovieLens 20M dataset from the given path.

        Args:
            path: Path to the dataset csv file (rating.csv)
            subset_ratio: Ratio of the dataset to load. Defaults to 1.0.
        """
        self.path = path
        print(f"Loading dataset from {path}...")
        self.data = pd.read_csv(path)

        if subset_ratio < 1.0:
            self.data = self.data.sample(frac=subset_ratio, random_state=42)

        print(f"Dataset loaded. Shape: {self.data.shape}")

        user_ids = np.array(self.data["userId"].unique())
        item_ids = np.array(self.data["movieId"].unique())

        # Reindexing user and item IDs to start from 0 and increment by 1
        self.item_id_map = {item_id: idx for idx, item_id in enumerate(item_ids)}
        self.item_id_reverse_map = {
            idx: item_id for item_id, idx in self.item_id_map.items()
        }
        user_id_map = {user_id: idx for idx, user_id in enumerate(user_ids)}

        self.data["movie_idx"] = self.data["movieId"].copy()

        self.data["userId"] = self.data["userId"].apply(lambda val: user_id_map[val])
        self.data["movieId"] = self.data["movieId"].apply(
            lambda val: self.item_id_map[val]
        )

        self.user_ids = np.array(self.data["userId"].unique())
        self.item_ids = np.array(self.data["movieId"].unique())

    def inject_user_row(self, ratings: Dict[int, float], increase_user_id: bool = False) -> None:
        """Inject a new user row into the dataset.

        Args:
            ratings: Dictionary containing the item IDs and ratings for the new user.
            {item_id: rating}
        """
        if increase_user_id:
            user_id = max(self.user_ids) + 1
        else:
            user_id = max(self.user_ids)

        item_ids = [self.item_id_map[item_id] for item_id in ratings.keys()]
        rating_values = list(ratings.values())

        new_rows = pd.DataFrame(
            {
                "userId": [user_id] * len(item_ids),
                "movieId": item_ids,
                "movie_idx": list(ratings.keys()),
                "rating": rating_values,
            }
        )

        self.data = pd.concat([self.data, new_rows], ignore_index=True)
        self.user_ids = np.append(self.user_ids, user_id)

    def get_train_test_split(
        self, test_size: float = 0.2, shuffle_set: bool = False
    ) -> Tuple["MovieLens20MDataset", "MovieLens20MDataset"]:
        """Split the dataset into train and test sets.

        Split the dataset is based on the user IDs.

        Args:
            test_size: Ratio of the dataset to be used for testing.
                    If set to 1.0, the entire dataset will be used for testing.
                    Thus, the train set will be empty.
            shuffle_set: Whether to shuffle the dataset before splitting.
                        If True, users will be randomly assigned to train and test sets.

        Returns:
            Tuple containing the train and test datasets.
            (train_dataset, test_dataset)
        """
        if test_size == 1.0:
            return MovieLens20MDataset(self.data.sample(0)), MovieLens20MDataset(
                self.data
            )

        if shuffle_set:
            train_data = self.data.sample(frac=1 - test_size, random_state=42)
            test_data = self.data.drop(train_data.index)  # type: ignore
        else:
            user_ids = self.user_ids.copy()
            n_train_ids = int(len(user_ids) * (1 - test_size))
            train_ids = user_ids[:n_train_ids]
            test_ids = user_ids[n_train_ids:]

            train_data = self.data[self.data["userId"].isin(train_ids)]  # type: ignore
            test_data = self.data[self.data["userId"].isin(test_ids)]  # type: ignore

        return MovieLens20MDataset(train_data), MovieLens20MDataset(test_data)  # type: ignore


class MovieLens20MDataset:
    """Class to represent the MovieLens 20M dataset."""

    def __init__(self, data: pd.DataFrame) -> None:
        """Initialize the dataset with the given data.

        Args:
            data: DataFrame containing the dataset.
        """
        self.data = data
        self.user_ids = data["userId"].unique()
        self.item_ids = data["movieId"].unique()

    def get_user_data(self, user_id: int) -> pd.DataFrame:
        """Get the data for the given user ID.

        Args:
            user_id: ID of the user.
        """
        return self.data[self.data["userId"] == user_id]  # type: ignore

    def get_item_data(self, item_ids: Union[int, List]) -> pd.DataFrame:
        """Get the data for the given item IDs.

        Args:
            item_ids: ID of the item(s).
                If a list is provided, data for all the items will be returned.
        """
        if isinstance(item_ids, int):
            item_ids = [item_ids]

        return self.data.query(f"movieId in {item_ids}")  # type: ignore


if __name__ == "__main__":
    path = "~/Datasets/MovieLens20M/rating.csv"

    dataset = MovieLens20MDatasetLoader(path)
    train_set, test_set = dataset.get_train_test_split(test_size=0.2, shuffle_set=True)
    data = train_set.get_user_data(train_set.user_ids[0])
    __import__("pdb").set_trace()
