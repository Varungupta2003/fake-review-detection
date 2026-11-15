from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments


@dataclass
class BertArtifacts:
    tokenizer: Any
    model: Any


def load_bert_artifacts(model_name: str = "bert-base-uncased") -> BertArtifacts:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    return BertArtifacts(tokenizer=tokenizer, model=model)


def tokenize_texts(tokenizer, texts, max_length: int = 256):
    return tokenizer(
        list(texts),
        truncation=True,
        padding=True,
        max_length=max_length,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
    preds = np.argmax(logits, axis=1)
    return {
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "pr_auc": float(average_precision_score(labels, probs)),
    }


def build_training_arguments(output_dir: str, epochs: int, batch_size: int, learning_rate: float):
    signature = inspect.signature(TrainingArguments.__init__)
    kwargs = {
        "output_dir": output_dir,
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "learning_rate": learning_rate,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "logging_strategy": "epoch",
    }

    if "evaluation_strategy" in signature.parameters:
        kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "epoch"

    if "report_to" in signature.parameters:
        kwargs["report_to"] = "none"

    return TrainingArguments(**kwargs)
