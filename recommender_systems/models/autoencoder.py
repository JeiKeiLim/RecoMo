from typing import Dict

import torch
import torch.nn as nn


class TorchAutoEncoderModel(nn.Module):
    def __init__(self, n_items: int, n_hidden: int, global_mean: float) -> None:
        super().__init__()
        self.n_items = n_items
        self.layer1 = nn.Linear(n_items, n_hidden)
        self.activation = nn.Tanh()
        self.layer2 = nn.Linear(n_hidden, n_items)
        self.dropout = nn.Dropout(0.7)
        self.global_mean = global_mean

    def forward(self, x: torch.Tensor, is_train: bool = False) -> torch.Tensor:
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
