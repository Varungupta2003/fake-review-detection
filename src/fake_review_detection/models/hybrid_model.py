from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class HybridBertStylometricClassifier(nn.Module):
    def __init__(
        self,
        bert_model_name: str,
        num_stylometric_features: int,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        bert_hidden = self.bert.config.hidden_size  # 768 for bert-base
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(bert_hidden + num_stylometric_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, input_ids, attention_mask, stylometric_features, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        fused = torch.cat([cls_embedding, stylometric_features], dim=1)
        logits = self.classifier(self.dropout(fused))

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}
