"""Visualize image, cell probabilities, ground truth and prediction."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from torch.utils.data import Subset

from configs import PRESETS
from dataset import ISBIDataset
from metrics import predict_probs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--preset", choices=PRESETS, default="paper_faithful")
    parser.add_argument("--slice", type=int, default=25)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--tta", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    test_ds = ISBIDataset(
        "ISBI-2012-challenge/test-volume.tif",
        "ISBI-2012-challenge/test-labels.tif",
    )
    image, label, _ = test_ds[args.slice]
    probs = predict_probs(
        args.checkpoint,
        Subset(test_ds, [args.slice]),
        padding=PRESETS[args.preset]["padding"],
        tta=args.tta,
    )[0]

    gt_seg, n_gt = ndimage.label(label.numpy() == 0)
    pred_seg, n_pred = ndimage.label(probs > args.threshold)
    print(f"slice {args.slice}: ground truth={n_gt} cells; prediction={n_pred} cells")

    rng = np.random.default_rng(0)
    colors = rng.random((max(n_gt, n_pred) + 1, 3))
    colors[0] = 0
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(image[0], cmap="gray")
    axes[0].set_title("Image")
    axes[1].imshow(probs, cmap="gray")
    axes[1].set_title("Cell probability")
    axes[2].imshow(colors[gt_seg])
    axes[2].set_title(f"Ground truth: {n_gt} cells")
    axes[3].imshow(colors[pred_seg])
    axes[3].set_title(f"Prediction: {n_pred} cells")
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
