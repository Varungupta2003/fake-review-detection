from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from fake_review_detection.utils import current_memory_mb


def evaluate_hybrid_model(model, dataloader, device):
    model.eval()
    all_labels = []
    all_preds = []
    all_scores = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            stylometric_features = batch["stylometric_features"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                stylometric_features=stylometric_features,
            )
            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=1)[:, 1]

            preds = torch.argmax(logits, dim=1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_scores.extend(probs.cpu().numpy().tolist())

    return {
        "f1": float(f1_score(all_labels, all_preds, zero_division=0)),
        "pr_auc": float(average_precision_score(all_labels, all_scores)),
        "y_true": np.array(all_labels),
        "y_pred": np.array(all_preds),
        "y_score": np.array(all_scores),
    }


def train_hybrid_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs: int = 3,
    learning_rate: float = 2e-5,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    best_state = None
    best_val_f1 = -1.0

    start_memory = current_memory_mb()
    train_start = time.perf_counter()

    model.to(device)

    for epoch in range(epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"Hybrid epoch {epoch+1}/{epochs}")
        for batch in progress:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            stylometric_features = batch["stylometric_features"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                stylometric_features=stylometric_features,
                labels=labels,
            )
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()

            progress.set_postfix(loss=float(loss.item()))

        val_metrics = evaluate_hybrid_model(model, val_loader, device)
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    train_time = time.perf_counter() - train_start
    peak_memory_delta_mb = max(0.0, current_memory_mb() - start_memory)

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "model": model,
        "training_time_seconds": train_time,
        "memory_delta_mb": peak_memory_delta_mb,
        "best_val_f1": best_val_f1,
    }


def measure_inference_latency(model, dataloader, device):
    model.eval()
    timings = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            stylometric_features = batch["stylometric_features"].to(device)

            start = time.perf_counter()
            _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                stylometric_features=stylometric_features,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings.append(time.perf_counter() - start)
    return float(np.mean(timings)) if timings else 0.0
