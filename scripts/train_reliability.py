#!/usr/bin/env python
"""Train TinyReliabilityNet from reliability .npz samples."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover - exercised only without the ml extra.
    raise RuntimeError('Torch is required. Install with: pip install -e ".[ml]"') from exc

from scapev3.learned.reliability import TinyReliabilityNet, save_reliability_checkpoint  # noqa: E402


class ReliabilitySampleDataset(Dataset):
    """Loads .npz samples written by build_reliability_dataset.py."""

    def __init__(self, labels_dirs: str | Path | list[str | Path]) -> None:
        if isinstance(labels_dirs, str | Path):
            labels_dirs = [labels_dirs]
        self.paths = []
        for labels_dir in labels_dirs:
            self.paths.extend(sorted(Path(labels_dir).glob("sample_*.npz")))
        if not self.paths:
            raise RuntimeError(f"No reliability samples found in: {labels_dirs}")
        self.positive_pixels, self.negative_pixels = self._count_labels()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        data = np.load(self.paths[index])
        return {
            "features": torch.from_numpy(data["features"].astype(np.float32)),
            "labels": torch.from_numpy(data["labels"][None, ...].astype(np.float32)),
            "mask": torch.from_numpy(data["mask"][None, ...].astype(np.float32)),
        }

    def _count_labels(self) -> tuple[int, int]:
        positives = 0
        negatives = 0
        for path in self.paths:
            data = np.load(path)
            labels = data["labels"].astype(bool)
            mask = data["mask"].astype(bool)
            positives += int((labels & mask).sum())
            negatives += int((~labels & mask).sum())
        return positives, negatives


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the 3DScape depth reliability model.")
    parser.add_argument("--labels-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--out-checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focal-loss gamma; 0 behaves like class-balanced BCE.",
    )
    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if not 0.0 <= args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be in [0, 1)")
    dataset = ReliabilitySampleDataset(args.labels_dir)
    train_dataset, val_dataset = _split_dataset(dataset, val_fraction=args.val_fraction, seed=args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False) if len(val_dataset) else None
    model = TinyReliabilityNet(base_channels=args.base_channels).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    class_weights = _balanced_class_weights(
        positives=dataset.positive_pixels,
        negatives=dataset.negative_pixels,
        device=args.device,
    )
    if val_loader is not None:
        heuristic_metrics = _evaluate_feature_heuristic(val_loader, device=args.device)
        print(_format_metrics("depth_edge_heuristic", heuristic_metrics))

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            features = batch["features"].to(args.device)
            labels = batch["labels"].to(args.device)
            mask = batch["mask"].to(args.device)
            logits = model(features)
            loss = _masked_focal_loss(
                logits=logits,
                labels=labels,
                mask=mask,
                class_weights=class_weights,
                gamma=args.focal_gamma,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        message = f"epoch {epoch:03d} train_loss {np.mean(losses):.5f}"
        if val_loader is not None:
            val_metrics = _evaluate(
                model,
                val_loader,
                class_weights=class_weights,
                gamma=args.focal_gamma,
                device=args.device,
            )
            message += (
                f" val_loss {val_metrics['loss']:.5f}"
                f" val_bal_acc {val_metrics['balanced_accuracy']:.4f}"
                f" val_f1 {val_metrics['f1']:.4f}"
                f" val_brier {val_metrics['brier']:.4f}"
                f" val_pos_rate {val_metrics['label_positive_rate']:.4f}"
            )
        print(message)

    checkpoint = save_reliability_checkpoint(
        args.out_checkpoint,
        model=model,
        extra={
            "training": {
                "labels_dir": [str(path) for path in args.labels_dir],
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "sample_count": len(dataset),
                "train_count": len(train_dataset),
                "val_count": len(val_dataset),
                "positive_pixels": dataset.positive_pixels,
                "negative_pixels": dataset.negative_pixels,
                "loss": "class_balanced_focal",
                "focal_gamma": args.focal_gamma,
            }
        },
    )
    print(f"checkpoint: {checkpoint}")


def _split_dataset(dataset: ReliabilitySampleDataset, *, val_fraction: float, seed: int):
    if val_fraction <= 0.0 or len(dataset) < 2:
        return dataset, torch.utils.data.Subset(dataset, [])
    generator = torch.Generator().manual_seed(seed)
    val_count = max(1, int(round(len(dataset) * val_fraction)))
    train_count = max(1, len(dataset) - val_count)
    if train_count + val_count > len(dataset):
        val_count = len(dataset) - train_count
    return torch.utils.data.random_split(dataset, [train_count, val_count], generator=generator)


def _balanced_class_weights(*, positives: int, negatives: int, device: str) -> torch.Tensor:
    total = max(positives + negatives, 1)
    positive_weight = total / max(2.0 * positives, 1.0)
    negative_weight = total / max(2.0 * negatives, 1.0)
    return torch.tensor([negative_weight, positive_weight], dtype=torch.float32, device=device)


def _masked_focal_loss(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    class_weights: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    ce = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    probabilities = torch.sigmoid(logits)
    pt = torch.where(labels > 0.5, probabilities, 1.0 - probabilities)
    weights = torch.where(labels > 0.5, class_weights[1], class_weights[0])
    focal = torch.pow((1.0 - pt).clamp_min(1e-6), gamma)
    loss_map = ce * weights * focal
    return (loss_map * mask).sum() / mask.sum().clamp_min(1.0)


def _evaluate(
    model: TinyReliabilityNet,
    loader: DataLoader,
    *,
    class_weights: torch.Tensor,
    gamma: float,
    device: str,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    stats = _empty_metric_stats()
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["mask"].to(device)
            logits = model(features)
            loss = _masked_focal_loss(
                logits=logits,
                labels=labels,
                mask=mask,
                class_weights=class_weights,
                gamma=gamma,
            )
            _update_metric_stats(stats, probabilities=torch.sigmoid(logits), labels=labels, mask=mask)
            losses.append(float(loss.detach().cpu()))
    metrics = _finish_metric_stats(stats)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    return metrics


def _evaluate_feature_heuristic(loader: DataLoader, *, device: str) -> dict[str, float]:
    """Evaluate a hand baseline that distrusts invalid/depth-edge pixels."""

    stats = _empty_metric_stats()
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["mask"].to(device)
            valid_depth = features[:, 4:5].clamp(0.0, 1.0)
            grad_x = features[:, 5:6]
            grad_y = features[:, 6:7]
            gradient_magnitude = torch.sqrt(grad_x.square() + grad_y.square())
            probabilities = valid_depth * (1.0 - torch.clamp(gradient_magnitude * 4.0, 0.0, 1.0))
            _update_metric_stats(stats, probabilities=probabilities, labels=labels, mask=mask)
    return _finish_metric_stats(stats)


def _empty_metric_stats() -> dict[str, float]:
    return {
        "total": 0.0,
        "tp": 0.0,
        "tn": 0.0,
        "fp": 0.0,
        "fn": 0.0,
        "label_positive": 0.0,
        "predicted_positive": 0.0,
        "brier_sum": 0.0,
        "mae_sum": 0.0,
    }


def _update_metric_stats(
    stats: dict[str, float],
    *,
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    valid = mask > 0
    if not bool(valid.any()):
        return
    valid_probabilities = probabilities[valid].clamp(0.0, 1.0)
    valid_labels = labels[valid].clamp(0.0, 1.0)
    hard_labels = valid_labels >= 0.5
    predictions = valid_probabilities >= 0.5
    stats["total"] += float(valid_labels.numel())
    stats["tp"] += float((predictions & hard_labels).sum().detach().cpu())
    stats["tn"] += float((~predictions & ~hard_labels).sum().detach().cpu())
    stats["fp"] += float((predictions & ~hard_labels).sum().detach().cpu())
    stats["fn"] += float((~predictions & hard_labels).sum().detach().cpu())
    stats["label_positive"] += float(hard_labels.sum().detach().cpu())
    stats["predicted_positive"] += float(predictions.sum().detach().cpu())
    stats["brier_sum"] += float((valid_probabilities - valid_labels).square().sum().detach().cpu())
    stats["mae_sum"] += float(torch.abs(valid_probabilities - valid_labels).sum().detach().cpu())


def _finish_metric_stats(stats: dict[str, float]) -> dict[str, float]:
    total = max(stats["total"], 1.0)
    tp = stats["tp"]
    tn = stats["tn"]
    fp = stats["fp"]
    fn = stats["fn"]
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    specificity = tn / max(tn + fp, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    return {
        "accuracy": (tp + tn) / total,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "brier": stats["brier_sum"] / total,
        "mae": stats["mae_sum"] / total,
        "label_positive_rate": stats["label_positive"] / total,
        "predicted_positive_rate": stats["predicted_positive"] / total,
    }


def _format_metrics(prefix: str, metrics: dict[str, float]) -> str:
    return (
        f"{prefix}"
        f" bal_acc {metrics['balanced_accuracy']:.4f}"
        f" f1 {metrics['f1']:.4f}"
        f" precision {metrics['precision']:.4f}"
        f" recall {metrics['recall']:.4f}"
        f" brier {metrics['brier']:.4f}"
    )


if __name__ == "__main__":
    main()
