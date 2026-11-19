"""Compare the custom Rand counts with scikit-image on one test slice."""

import argparse

import numpy as np
from scipy import ndimage
from skimage.metrics import adapted_rand_error
from torch.utils.data import Subset

from configs import PRESETS
from dataset import ISBIDataset
from metrics import predict_probs, rand_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--preset", choices=PRESETS, default="paper_faithful")
    parser.add_argument("--slice", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    test_ds = ISBIDataset(
        "ISBI-2012-challenge/test-volume.tif",
        "ISBI-2012-challenge/test-labels.tif",
    )
    one = Subset(test_ds, [args.slice])
    probs = predict_probs(
        args.checkpoint,
        one,
        padding=PRESETS[args.preset]["padding"],
    )[0]
    gt_cell = test_ds[args.slice][1].numpy() == 0
    pred_cell = probs > args.threshold

    s_both, s_gt, s_pred = rand_counts(pred_cell, gt_cell)
    print(f"custom:          P={s_both / s_pred:.6f} R={s_both / s_gt:.6f}")

    gt_seg, _ = ndimage.label(gt_cell)
    pred_seg, _ = ndimage.label(pred_cell)
    _, precision, recall = adapted_rand_error(gt_seg, pred_seg)
    print(f"scikit-image A: P={recall:.6f} R={precision:.6f}")

    pred_single = pred_seg.copy()
    zero = pred_single == 0
    pred_single[zero] = np.arange(zero.sum()) + pred_seg.max() + 1
    _, precision, recall = adapted_rand_error(gt_seg, pred_single)
    print(f"scikit-image B: P={recall:.6f} R={precision:.6f}")


if __name__ == "__main__":
    main()
