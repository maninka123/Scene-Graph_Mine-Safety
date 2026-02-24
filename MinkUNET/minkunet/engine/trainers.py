from __future__ import annotations

import csv
import json
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from ..losses.contrastive import nt_xent_loss
from ..models.network import sparse_tensor_from_batch
from ..utils.metrics import compute_segmentation_metrics, confusion_matrix
from .checkpoint import save_checkpoint
from .eval import evaluate_contrastive, evaluate_segmentation

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional at runtime
    plt = None


def _init_run_dirs(cfg: Dict, phase: str) -> Dict[str, Path]:
    logging_cfg = cfg.get("logging", {})
    root_dir = Path(logging_cfg.get("root_dir", "MinkUNET/logs"))
    custom_name = str(logging_cfg.get("run_name", "")).strip()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{custom_name}_{timestamp}" if custom_name else timestamp

    run_dir = root_dir / phase / run_name
    metrics_dir = run_dir / "metrics"
    plots_dir = run_dir / "plots"
    artifacts_dir = run_dir / "artifacts"
    diagnostics_dir = run_dir / "diagnostics"
    for d in (run_dir, metrics_dir, plots_dir, artifacts_dir, diagnostics_dir):
        d.mkdir(parents=True, exist_ok=True)

    with (artifacts_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(cfg, handle, indent=2)

    return {
        "run_dir": run_dir,
        "metrics_dir": metrics_dir,
        "plots_dir": plots_dir,
        "artifacts_dir": artifacts_dir,
        "diagnostics_dir": diagnostics_dir,
    }


def _append_jsonl(path: Path, row: Dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def _append_csv(path: Path, row: Dict) -> None:
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _plot_lines(out_path: Path, x_vals, series: Dict[str, list[float]], title: str, y_label: str) -> None:
    if plt is None:
        return
    plt.figure(figsize=(8, 5))
    for label, values in series.items():
        plt.plot(x_vals, values, label=label)
    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    if len(series) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_similarity_histogram(out_path: Path, pos_vals: np.ndarray, neg_vals: np.ndarray, epoch: int) -> None:
    if plt is None:
        return
    plt.figure(figsize=(8, 5))
    bins = np.linspace(-1.0, 1.0, 60)
    if pos_vals.size:
        plt.hist(pos_vals, bins=bins, alpha=0.55, label="positive pairs", density=True)
    if neg_vals.size:
        plt.hist(neg_vals, bins=bins, alpha=0.55, label="negative pairs", density=True)
    plt.title(f"Contrastive Similarity Histogram (Epoch {epoch})")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def _plot_layer_sparsity(out_path: Path, layer_rows: list[Dict], epoch: int) -> None:
    if plt is None or not layer_rows:
        return
    layer_names = [r["layer"] for r in layer_rows]
    voxel_counts = [float(r["active_voxels_mean"]) for r in layer_rows]
    activation_means = [float(r["activation_abs_mean"]) for r in layer_rows]
    densities = [float(r["feature_density_mean"]) for r in layer_rows]

    x = np.arange(len(layer_names))
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].bar(x, voxel_counts)
    axes[0].set_ylabel("Active Voxels")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].set_title(f"Layer Sparsity/Activation (Epoch {epoch})")

    axes[1].bar(x, activation_means)
    axes[1].set_ylabel("Mean |Activation|")
    axes[1].grid(True, axis="y", alpha=0.3)

    axes[2].bar(x, densities)
    axes[2].set_ylabel("Feature Density")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].grid(True, axis="y", alpha=0.3)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(layer_names, rotation=35, ha="right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


class _BackboneLayerStatsRecorder:
    def __init__(self, model) -> None:
        self.model = model
        self.layer_names = ["stem", "enc1", "enc2", "enc3", "enc4", "dec3", "dec2", "dec1", "out_proj"]
        self._handles = []
        self._acc: Dict[str, Dict[str, float]] = {}
        self._register()
        self.reset_epoch()

    def _register(self) -> None:
        backbone = getattr(self.model, "backbone", None)
        if backbone is None:
            return
        for name in self.layer_names:
            if not hasattr(backbone, name):
                continue
            module = getattr(backbone, name)
            handle = module.register_forward_hook(self._make_hook(name))
            self._handles.append(handle)

    def _make_hook(self, layer_name: str):
        def hook(_module, _inputs, output):
            if not hasattr(output, "F"):
                return
            feats = output.F.detach()
            if feats.numel() == 0:
                return
            acc = self._acc.setdefault(
                layer_name,
                {"calls": 0.0, "active_voxels_sum": 0.0, "activation_abs_sum": 0.0, "feature_density_sum": 0.0},
            )
            acc["calls"] += 1.0
            acc["active_voxels_sum"] += float(feats.shape[0])
            acc["activation_abs_sum"] += float(feats.abs().mean().item())
            density = float((feats.abs() > 1e-8).float().mean().item())
            acc["feature_density_sum"] += density

        return hook

    def reset_epoch(self) -> None:
        self._acc = {}

    def epoch_rows(self) -> list[Dict]:
        rows = []
        for name in self.layer_names:
            acc = self._acc.get(name)
            if not acc or acc["calls"] <= 0:
                continue
            c = acc["calls"]
            rows.append(
                {
                    "layer": name,
                    "active_voxels_mean": acc["active_voxels_sum"] / c,
                    "activation_abs_mean": acc["activation_abs_sum"] / c,
                    "feature_density_mean": acc["feature_density_sum"] / c,
                }
            )
        return rows

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


def _autocast_context(device: torch.device, enabled: bool):
    return torch.autocast(
        device_type="cuda" if device.type == "cuda" else "cpu",
        enabled=enabled and device.type == "cuda",
    )


def _make_grad_scaler(device: torch.device, enabled: bool):
    use_amp = bool(enabled and device.type == "cuda")
    try:
        return torch.amp.GradScaler(device="cuda", enabled=use_amp)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=use_amp)


def train_contrastive(
    model,
    train_loader,
    val_loader,
    cfg: Dict,
    device: torch.device,
) -> Path:
    train_cfg = cfg["train"]
    pretrain_cfg = cfg["pretrain"]
    checkpoint_dir = Path(cfg["checkpoint"]["dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_paths = _init_run_dirs(cfg, phase="contrastive")
    epoch_jsonl = run_paths["metrics_dir"] / "epoch_metrics.jsonl"
    epoch_csv = run_paths["metrics_dir"] / "epoch_metrics.csv"
    diagnostics_dir = run_paths["diagnostics_dir"]
    diag_metrics_jsonl = diagnostics_dir / "similarity_metrics.jsonl"
    diag_metrics_csv = diagnostics_dir / "similarity_metrics.csv"
    layer_metrics_jsonl = diagnostics_dir / "layer_sparsity_metrics.jsonl"
    layer_metrics_csv = diagnostics_dir / "layer_sparsity_metrics.csv"
    print(f"[Logs] Contrastive metrics will be saved to {run_paths['run_dir']}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(train_cfg["epochs"]))
    requested_amp = bool(train_cfg.get("amp", True))
    # MinkowskiEngine global pooling backward is not implemented for fp16 in 0.5.4.
    # Contrastive training uses global pooling, so force AMP off to avoid runtime failure.
    amp_enabled = False if device.type == "cuda" else requested_amp
    if requested_amp and device.type == "cuda":
        print("[Warn] Disabling AMP for contrastive pretraining due to MinkowskiEngine fp16 pooling limitation.")
    scaler = _make_grad_scaler(device=device, enabled=amp_enabled)

    temperature = float(pretrain_cfg["temperature"])
    best_val_loss = float("inf")
    best_checkpoint = checkpoint_dir / pretrain_cfg.get("checkpoint_name", "contrastive_best.pt")
    best_epoch = 0
    history = []
    run_start = time.perf_counter()
    recorder = _BackboneLayerStatsRecorder(model)
    diag_batches = int(pretrain_cfg.get("diagnostics_batches_per_epoch", 3))
    max_neg_samples = int(pretrain_cfg.get("diagnostics_max_negative_samples", 40000))

    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        epoch_start = time.perf_counter()
        model.train()
        running_loss = 0.0
        steps = 0
        recorder.reset_epoch()
        pos_sims_epoch: list[np.ndarray] = []
        neg_sims_epoch: list[np.ndarray] = []

        for batch in tqdm(train_loader, desc=f"Pretrain {epoch}", leave=False):
            optimizer.zero_grad(set_to_none=True)

            sparse_a = sparse_tensor_from_batch(batch["coords_a"], batch["features_a"], device=device)
            sparse_b = sparse_tensor_from_batch(batch["coords_b"], batch["features_b"], device=device)

            with _autocast_context(device=device, enabled=amp_enabled):
                emb_a, _ = model(sparse_a)
                emb_b, _ = model(sparse_b)
                loss = nt_xent_loss(emb_a, emb_b, temperature=temperature)

            if steps < diag_batches:
                with torch.no_grad():
                    pos = torch.sum(emb_a * emb_b, dim=1).detach().cpu().numpy()
                    cross = torch.matmul(emb_a, emb_b.T).detach().cpu().numpy()
                    mask = ~np.eye(cross.shape[0], dtype=bool)
                    neg = cross[mask]
                    if neg.size > max_neg_samples:
                        idx = np.random.choice(neg.size, size=max_neg_samples, replace=False)
                        neg = neg[idx]
                    pos_sims_epoch.append(pos.astype(np.float32))
                    neg_sims_epoch.append(neg.astype(np.float32))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_clip = float(train_cfg.get("grad_clip_norm", 0.0))
            if grad_clip > 0:
                clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item())
            steps += 1

        scheduler.step()
        train_loss = running_loss / max(steps, 1)
        if val_loader is not None:
            val_loss = evaluate_contrastive(model, val_loader, device=device, temperature=temperature)
        else:
            val_loss = train_loss
        lr = float(optimizer.param_groups[0]["lr"])
        epoch_time_sec = float(time.perf_counter() - epoch_start)

        checkpoint_state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "config": cfg,
        }

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            save_checkpoint(best_checkpoint, checkpoint_state)

        if epoch % int(cfg["checkpoint"].get("save_every", 5)) == 0:
            save_checkpoint(checkpoint_dir / f"contrastive_epoch_{epoch:03d}.pt", checkpoint_state)

        row = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "best_val_loss": float(best_val_loss),
            "lr": lr,
            "steps": int(steps),
            "epoch_time_sec": epoch_time_sec,
            "val_enabled": bool(val_loader is not None),
        }
        history.append(row)
        _append_jsonl(epoch_jsonl, row)
        _append_csv(epoch_csv, row)

        pos_vals = np.concatenate(pos_sims_epoch, axis=0) if pos_sims_epoch else np.zeros((0,), dtype=np.float32)
        neg_vals = np.concatenate(neg_sims_epoch, axis=0) if neg_sims_epoch else np.zeros((0,), dtype=np.float32)
        sim_row = {
            "epoch": int(epoch),
            "positive_mean": float(pos_vals.mean()) if pos_vals.size else 0.0,
            "positive_std": float(pos_vals.std()) if pos_vals.size else 0.0,
            "negative_mean": float(neg_vals.mean()) if neg_vals.size else 0.0,
            "negative_std": float(neg_vals.std()) if neg_vals.size else 0.0,
            "separation_margin": float((pos_vals.mean() - neg_vals.mean()) if (pos_vals.size and neg_vals.size) else 0.0),
            "positive_count": int(pos_vals.size),
            "negative_count": int(neg_vals.size),
        }
        _append_jsonl(diag_metrics_jsonl, sim_row)
        _append_csv(diag_metrics_csv, sim_row)
        _plot_similarity_histogram(
            diagnostics_dir / f"similarity_hist_epoch_{epoch:03d}.png",
            pos_vals=pos_vals,
            neg_vals=neg_vals,
            epoch=epoch,
        )

        layer_rows = recorder.epoch_rows()
        for lrw in layer_rows:
            layer_row = {
                "epoch": int(epoch),
                "layer": lrw["layer"],
                "active_voxels_mean": float(lrw["active_voxels_mean"]),
                "activation_abs_mean": float(lrw["activation_abs_mean"]),
                "feature_density_mean": float(lrw["feature_density_mean"]),
            }
            _append_jsonl(layer_metrics_jsonl, layer_row)
            _append_csv(layer_metrics_csv, layer_row)
        _plot_layer_sparsity(
            diagnostics_dir / f"layer_sparsity_epoch_{epoch:03d}.png",
            layer_rows=layer_rows,
            epoch=epoch,
        )

        print(
            f"[Contrastive] epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} best={best_val_loss:.4f}"
        )

    _plot_lines(
        run_paths["plots_dir"] / "loss_curve.png",
        [r["epoch"] for r in history],
        {"train_loss": [r["train_loss"] for r in history], "val_loss": [r["val_loss"] for r in history]},
        title="Contrastive Loss vs Epoch",
        y_label="NT-Xent Loss",
    )
    _plot_lines(
        run_paths["plots_dir"] / "lr_curve.png",
        [r["epoch"] for r in history],
        {"lr": [r["lr"] for r in history]},
        title="Learning Rate vs Epoch",
        y_label="Learning Rate",
    )

    summary = {
        "phase": "contrastive",
        "best_checkpoint": str(best_checkpoint),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "epochs": int(train_cfg["epochs"]),
        "total_time_sec": float(time.perf_counter() - run_start),
        "metrics_jsonl": str(epoch_jsonl),
        "metrics_csv": str(epoch_csv),
        "plots_dir": str(run_paths["plots_dir"]),
        "diagnostics_dir": str(diagnostics_dir),
        "diagnostic_similarity_csv": str(diag_metrics_csv),
        "diagnostic_layer_sparsity_csv": str(layer_metrics_csv),
    }
    with (run_paths["artifacts_dir"] / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    recorder.close()

    return best_checkpoint


def _load_pretrained_backbone(model, checkpoint_path: str | Path) -> None:
    p = Path(checkpoint_path)
    if not p.exists():
        print(f"[FineTune] Pretrained checkpoint not found, training from scratch: {p}")
        return

    state = torch.load(p, map_location="cpu")
    model_state = state.get("model_state", state)
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    print(
        f"[FineTune] Loaded pretrained weights from {p}. "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )


def train_segmentation(
    model,
    train_loader,
    val_loader,
    cfg: Dict,
    device: torch.device,
) -> Path:
    train_cfg = cfg["train"]
    ft_cfg = cfg["finetune"]
    model_cfg = cfg["model"]
    checkpoint_dir = Path(cfg["checkpoint"]["dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_paths = _init_run_dirs(cfg, phase="segmentation")
    epoch_jsonl = run_paths["metrics_dir"] / "epoch_metrics.jsonl"
    epoch_csv = run_paths["metrics_dir"] / "epoch_metrics.csv"
    print(f"[Logs] Segmentation metrics will be saved to {run_paths['run_dir']}")

    pretrained_ckpt = ft_cfg.get("pretrained_checkpoint")
    if pretrained_ckpt:
        _load_pretrained_backbone(model, pretrained_ckpt)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(train_cfg["epochs"]))
    scaler = _make_grad_scaler(device=device, enabled=bool(train_cfg.get("amp", True)))

    class_weights = torch.tensor(ft_cfg.get("class_weights", [1.0] * int(model_cfg["num_classes"])), dtype=torch.float32)
    class_weights = class_weights.to(device)
    ignore_index = int(ft_cfg.get("ignore_index", -1))
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)

    num_classes = int(model_cfg["num_classes"])
    class_names = model_cfg["class_names"]

    best_miou = -1.0
    best_checkpoint = checkpoint_dir / ft_cfg.get("checkpoint_name", "semantic_best.pt")
    best_epoch = 0
    best_val_metrics = {}
    history = []
    run_start = time.perf_counter()

    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        epoch_start = time.perf_counter()
        model.train()
        running_loss = 0.0
        steps = 0
        hist = np.zeros((num_classes, num_classes), dtype=np.int64)

        for batch in tqdm(train_loader, desc=f"FineTune {epoch}", leave=False):
            optimizer.zero_grad(set_to_none=True)

            sparse_input = sparse_tensor_from_batch(batch["coords"], batch["features"], device=device)
            labels = batch["labels"].to(device)

            with _autocast_context(device=device, enabled=bool(train_cfg.get("amp", True))):
                logits = model(sparse_input).F
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_clip = float(train_cfg.get("grad_clip_norm", 0.0))
            if grad_clip > 0:
                clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item())
            steps += 1

            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            targets = labels.detach().cpu().numpy()
            hist += confusion_matrix(preds, targets, num_classes=num_classes, ignore_index=ignore_index)

        scheduler.step()

        train_loss = running_loss / max(steps, 1)
        train_metrics = compute_segmentation_metrics(hist, class_names=class_names)
        val_metrics = evaluate_segmentation(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            class_names=class_names,
            ignore_index=ignore_index,
        )
        lr = float(optimizer.param_groups[0]["lr"])
        epoch_time_sec = float(time.perf_counter() - epoch_start)

        checkpoint_state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "train_loss": train_loss,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "config": cfg,
        }

        val_miou = float(val_metrics["miou"])
        if val_miou > best_miou:
            best_miou = val_miou
            best_epoch = epoch
            best_val_metrics = deepcopy(val_metrics)
            save_checkpoint(best_checkpoint, checkpoint_state)

        if epoch % int(cfg["checkpoint"].get("save_every", 5)) == 0:
            save_checkpoint(checkpoint_dir / f"semantic_epoch_{epoch:03d}.pt", checkpoint_state)

        row = {
            "epoch": int(epoch),
            "lr": lr,
            "steps": int(steps),
            "epoch_time_sec": epoch_time_sec,
            "train_loss": float(train_loss),
            "val_loss": float(val_metrics["loss"]),
            "train_miou": float(train_metrics["miou"]),
            "val_miou": float(val_metrics["miou"]),
            "train_accuracy": float(train_metrics["accuracy"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "best_val_miou": float(best_miou),
        }
        for class_name in class_names:
            row[f"train_iou_{class_name}"] = float(train_metrics["per_class"][class_name]["iou"])
            row[f"val_iou_{class_name}"] = float(val_metrics["per_class"][class_name]["iou"])

        history.append(row)
        _append_jsonl(epoch_jsonl, row)
        _append_csv(epoch_csv, row)

        print(
            f"[FineTune] epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} train_mIoU={train_metrics['miou']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_mIoU={val_metrics['miou']:.4f} "
            f"best_val_mIoU={best_miou:.4f}"
        )

    epochs = [r["epoch"] for r in history]
    _plot_lines(
        run_paths["plots_dir"] / "loss_curve.png",
        epochs,
        {"train_loss": [r["train_loss"] for r in history], "val_loss": [r["val_loss"] for r in history]},
        title="Segmentation Loss vs Epoch",
        y_label="Cross-Entropy Loss",
    )
    _plot_lines(
        run_paths["plots_dir"] / "miou_curve.png",
        epochs,
        {"train_mIoU": [r["train_miou"] for r in history], "val_mIoU": [r["val_miou"] for r in history]},
        title="mIoU vs Epoch",
        y_label="mIoU",
    )
    _plot_lines(
        run_paths["plots_dir"] / "accuracy_curve.png",
        epochs,
        {"train_accuracy": [r["train_accuracy"] for r in history], "val_accuracy": [r["val_accuracy"] for r in history]},
        title="Accuracy vs Epoch",
        y_label="Accuracy",
    )
    _plot_lines(
        run_paths["plots_dir"] / "lr_curve.png",
        epochs,
        {"lr": [r["lr"] for r in history]},
        title="Learning Rate vs Epoch",
        y_label="Learning Rate",
    )

    per_class_series = {f"val_iou_{name}": [r[f"val_iou_{name}"] for r in history] for name in class_names}
    _plot_lines(
        run_paths["plots_dir"] / "val_per_class_iou.png",
        epochs,
        per_class_series,
        title="Validation Per-Class IoU vs Epoch",
        y_label="IoU",
    )

    summary = {
        "phase": "segmentation",
        "best_checkpoint": str(best_checkpoint),
        "best_epoch": int(best_epoch),
        "best_val_miou": float(best_miou),
        "best_val_metrics": best_val_metrics,
        "epochs": int(train_cfg["epochs"]),
        "total_time_sec": float(time.perf_counter() - run_start),
        "metrics_jsonl": str(epoch_jsonl),
        "metrics_csv": str(epoch_csv),
        "plots_dir": str(run_paths["plots_dir"]),
    }
    with (run_paths["artifacts_dir"] / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return best_checkpoint
