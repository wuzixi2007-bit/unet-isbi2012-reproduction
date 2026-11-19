"""Evaluate one checkpoint with the challenge-style pixel and Rand errors."""

import argparse

import numpy as np

from configs import PRESETS
from dataset import ISBIDataset
from metrics import pixel_error, predict_probs, rand_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="path to a .pth state_dict checkpoint")
    parser.add_argument(
        "--preset",
        choices=PRESETS,
        default="best_empirical",
        help="architecture preset used to train the checkpoint",
    )
    parser.add_argument(
        "--tta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use eight-way rotation/flip test-time augmentation (default: true)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PRESETS[args.preset]
    test_ds = ISBIDataset(
        "ISBI-2012-challenge/test-volume.tif",
        "ISBI-2012-challenge/test-labels.tif",
    )
    ground_truth_cells = np.stack([test_ds[i][1].numpy() == 0 for i in range(len(test_ds))])
    probabilities = predict_probs(
        args.checkpoint,
        test_ds,
        padding=config["padding"],
        tta=args.tta,
    )

    print(f"checkpoint: {args.checkpoint}")
    print(f"preset: {args.preset}; TTA: {args.tta}; images: {len(test_ds)}")
    print(f"pixel error: {pixel_error(probabilities, ground_truth_cells):.4f}")
    print(f"Rand error:  {rand_error(probabilities, ground_truth_cells):.4f}")


if __name__ == "__main__":
    main()
