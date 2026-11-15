from __future__ import annotations

import torch
from torch.utils.data import Dataset


class HybridReviewDataset(Dataset):
    def __init__(self, encodings, stylometric_features, labels):
        self.encodings = encodings
        self.stylometric_features = stylometric_features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["stylometric_features"] = torch.tensor(self.stylometric_features[idx], dtype=torch.float32)
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
