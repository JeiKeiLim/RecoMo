from typing import Dict

import torch
import torch.nn as nn
from models.autoencoder import TorchAutoEncoderModel
from models.matrix_factorization import TorchMatrixFactorizationModel
from trainer.dataset_loader import MovieLens20MDatasetLoader


class ModelWrapper:
    def __init__(self, model: nn.Module, dataset_path: str) -> None:
        self.model = model
        self.dataset = MovieLens20MDatasetLoader(dataset_path, subset_ratio=1.0)
        self.did_inject_user_row = False

    def _predict_matrix_factorization(self, data: Dict[int, float]) -> Dict[int, float]:
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
        idx_map = self.dataset.item_id_map
        predictions = self.model.predict(data, idx_map)

        return predictions

    @torch.no_grad()
    def predict(self, data: Dict[int, float]) -> Dict[int, float]:
        if isinstance(self.model, TorchMatrixFactorizationModel):
            return self._predict_matrix_factorization(data)
        elif isinstance(self.model, TorchAutoEncoderModel):
            return self._predict_autoencoder(data)


def model_loader(name: str, config: Dict, device: torch.device) -> nn.Module:
    model_dict = {
        "matrix_factorization": TorchMatrixFactorizationModel,
        "autoencoder": TorchAutoEncoderModel,
    }

    if name not in model_dict:
        raise ValueError(f"Model {name} not found in model_dict")

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
