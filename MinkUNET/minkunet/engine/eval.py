from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from tqdm import tqdm

from ..losses.contrastive import nt_xent_loss
from ..models.network import sparse_tensor_from_batch
from ..utils.metrics import compute_segmentation_metrics, confusion_matrix


@torch.no_grad()
def evaluate_contrastive(model, loader, device: torch.device, temperature: float) -> float:
    model.eval()
    losses = []
    for batch in tqdm(loader, desc="Val Contrastive", leave=False):
        sparse_a = sparse_tensor_from_batch(batch["coords_a"], batch["features_a"], device=device)
        sparse_b = sparse_tensor_from_batch(batch["coords_b"], batch["features_b"], device=device)
        emb_a, _ = model(sparse_a)
        emb_b, _ = model(sparse_b)
        loss = nt_xent_loss(emb_a, emb_b, temperature=temperature)
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def evaluate_segmentation(
    model,
    loader,
    criterion,
    device: torch.device,
    num_classes: int,
    class_names,
    ignore_index: int,
    return_hist: bool = False,
) -> Dict:
    model.eval()
    total_loss = 0.0
    total_steps = 0
    hist = np.zeros((num_classes, num_classes), dtype=np.int64)

    for batch in tqdm(loader, desc="Val Segmentation", leave=False):
        sparse_input = sparse_tensor_from_batch(batch["coords"], batch["features"], device=device)
        labels = batch["labels"].to(device)

        logits = model(sparse_input).F
        loss = criterion(logits, labels)
        total_loss += float(loss.item())
        total_steps += 1

        predictions = torch.argmax(logits, dim=1).detach().cpu().numpy()
        targets = labels.detach().cpu().numpy()
        hist += confusion_matrix(predictions, targets, num_classes=num_classes, ignore_index=ignore_index)

    metrics = compute_segmentation_metrics(hist, class_names=class_names)
    metrics["loss"] = total_loss / max(total_steps, 1)
    if return_hist:
        metrics["confusion_matrix"] = hist.astype(np.int64)
    return metrics
