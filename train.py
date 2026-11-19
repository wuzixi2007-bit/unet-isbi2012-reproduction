"""Train either the best empirical or the paper-like U-Net configuration."""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from configs import BATCH_SIZE, LEARNING_RATE, MOMENTUM, PRESETS
from dataset import ISBIDataset, load_tif_stack
from model import UNet
from tile import predict_tiled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=PRESETS,
        default="best_empirical",
        help="named experiment configuration (default: best_empirical)",
    )
    parser.add_argument("--seed", type=int, default=42, help="training seed (default: 42)")
    parser.add_argument("--epochs", type=int, default=300, help="number of epochs (default: 300)")
    parser.add_argument("--run-name", help="checkpoint name prefix; derived from preset and seed by default")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="training device (default: auto)",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    args = parse_args()
    config = PRESETS[args.preset]
    padding = config["padding"]
    dropout = config["dropout"]
    use_weights = config["use_weights"]
    run_name = args.run_name or f"{args.preset}_seed{args.seed}"
    device = resolve_device(args.device)

    torch.manual_seed(args.seed)
    indices = torch.randperm(30, generator=torch.Generator().manual_seed(42)).tolist()

    print(f"preset: {args.preset} ({config['description']})")
    print(f"device: {device}; seed: {args.seed}; validation indices: {indices[25:]}")
    print(
        f"lr={LEARNING_RATE}, momentum={MOMENTUM}, batch_size={BATCH_SIZE}, "
        f"padding={padding}, dropout={dropout}, weighted_loss={use_weights}"
    )

    class_weights = None
    if use_weights:
        all_labels = load_tif_stack("ISBI-2012-challenge/train-labels.tif")
        membrane_fraction = (all_labels[indices[:25]] == 0).mean()
        class_weights = {
            0: 1 / (2 * (1 - membrane_fraction)),
            1: 1 / (2 * membrane_fraction),
        }
        print(
            f"membrane fraction={membrane_fraction:.3f}; "
            f"class weights={{0: {class_weights[0]:.3f}, 1: {class_weights[1]:.3f}}}"
        )

    train_full = ISBIDataset(
        "ISBI-2012-challenge/train-volume.tif",
        "ISBI-2012-challenge/train-labels.tif",
        augment=True,
        elastic=True,
        wc=class_weights,
        crop=(padding == 0),
    )
    val_full = ISBIDataset(
        "ISBI-2012-challenge/train-volume.tif",
        "ISBI-2012-challenge/train-labels.tif",
        augment=False,
        elastic=False,
    )
    train_ds = Subset(train_full, indices[:25])
    val_ds = Subset(val_full, indices[25:])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = UNet(padding=padding, dropout=dropout).to(device)
    pixel_loss_fn = nn.CrossEntropyLoss(reduction="none")
    val_loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    best_val_loss = float("inf")
    best_val_f1 = 0.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        started = time.time()

        for images, labels, weights in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            weights = weights.to(device)
            pixel_loss = pixel_loss_fn(model(images), labels)
            loss = (pixel_loss * weights).sum() / weights.sum() if use_weights else pixel_loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct = total = tp = fp = fn = 0
        with torch.no_grad():
            for image, label, _ in val_ds:
                if padding == 0:
                    logits = predict_tiled(model, image)[None]
                else:
                    logits = model(image[None].to(device)).cpu()

                labels = label[None]
                val_loss += val_loss_fn(logits, labels).item()
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.numel()
                tp += ((preds == 1) & (labels == 1)).sum().item()
                fp += ((preds == 1) & (labels == 0)).sum().item()
                fn += ((preds == 0) & (labels == 1)).sum().item()

        avg_loss = total_loss / len(train_loader)
        val_loss /= len(val_ds)
        val_acc = correct / total
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        print(
            f"epoch {epoch + 1}/{args.epochs} train_loss={avg_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"P={precision:.4f} R={recall:.4f} F1={f1:.4f} "
            f"time={time.time() - started:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_dir / f"{run_name}_best_loss.pth")
            print("  saved new best validation-loss checkpoint")

        if f1 > best_val_f1:
            best_val_f1 = f1
            torch.save(model.state_dict(), checkpoint_dir / f"{run_name}_best_f1.pth")
            print("  saved new best validation-F1 checkpoint")

    print(f"finished: best val_loss={best_val_loss:.4f}; best val_F1={best_val_f1:.4f}")


if __name__ == "__main__":
    main()
