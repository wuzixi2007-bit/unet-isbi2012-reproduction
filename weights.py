import time

import numpy as np
from scipy import ndimage


def compute_weight_map(label, wc, w0=10.0, sigma=5.0):
    """
    论文第 8 题的权重图。
    label: (H, W) numpy 数组，1 = 细胞膜，0 = 细胞内部
    wc:    字典 {0: 细胞内部的权重, 1: 细胞膜的权重}
    """
    H, W = label.shape

    # ① 类别权重：细胞膜像素用 wc[1]，细胞内部像素用 wc[0]
    weight = np.where(label == 1, wc[1], wc[0]).astype(np.float32)

    # ② 给每个细胞编号：细胞内部的每一块连通区域就是一个细胞
    cells, n = ndimage.label(label == 0)
    if n < 2:                                   # 少于 2 个细胞，没有 d2，只用类别权重
        return weight

    # ③ 每个像素到每个细胞的距离 → (n, H, W)
    dists = np.zeros((n, H, W), dtype=np.float32)
    for i in range(1, n + 1):
        dists[i - 1] = ndimage.distance_transform_edt(cells != i)

    # ④ 取最近的两个距离
    dists = np.partition(dists, 1, axis=0)      # 把最小的两个放到最前面
    d1, d2 = dists[0], dists[1]

    # ⑤ 边界权重，只加在细胞膜像素上
    border = w0 * np.exp(-((d1 + d2) ** 2) / (2 * sigma ** 2))
    weight += border * (label == 1)

    return weight


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from dataset import load_tif_stack

    labels = load_tif_stack("ISBI-2012-challenge/train-labels.tif")
    labels = (labels == 0).astype(np.int64)     # 和 Dataset 里一样：细胞膜 = 1

    # 用整个训练集计算类别频率
    f1 = labels.mean()                          # 细胞膜的比例
    wc = {0: 1 / (2 * (1 - f1)), 1:1 / (2 * f1)}
    print(f"细胞膜比例 {f1:.3f}  →  wc = {{0: {wc[0]:.3f}, 1: {wc[1]:.3f}}}")

    label = labels[0]
    start = time.time()
    weight = compute_weight_map(label, wc)
    print(f"计算一张权重图用时 {time.time() - start:.2f} 秒")
    print(f"权重范围: {weight.min():.3f} ~ {weight.max():.3f}，平均 {weight.mean():.3f}")

    cells, n = ndimage.label(label == 0)
    print("细胞个数:", n)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(label, cmap="gray")
    axes[0].set_title("Label (membrane = white)")
    axes[1].imshow(cells, cmap="tab20")
    axes[1].set_title(f"Cells ({n} instances)")
    im = axes[2].imshow(weight, cmap="jet")
    axes[2].set_title("Weight map")
    plt.colorbar(im, ax=axes[2])
    plt.show()