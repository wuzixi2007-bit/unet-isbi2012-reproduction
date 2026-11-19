import numpy as np
import torch
from scipy import ndimage

from dataset import ISBIDataset
from model import UNet
from tile import predict_tiled

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

def predict_one(model, image, padding):
    """image: (1, H, W) → 细胞内部的概率图 (H, W)，在 CPU 上"""
    if padding == 0:
        logits = predict_tiled(model, image)[None]
    else:
        logits = model(image.unsqueeze(0).to(device))
    return torch.softmax(logits, dim=1)[0, 0].cpu()

# def predict_probs(model_path, ds, padding=1):
#     """对 ds 里每张图，返回"细胞内部"的概率图 → (N, H, W)"""
#     model = UNet(padding=padding).to(device)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.eval()
#
#     probs = []
#     with torch.no_grad():
#         for i in range(len(ds)):
#             image, _, _ = ds[i]
#             if padding == 0:
#                 logits = predict_tiled(model, image)[None]                 # (1, 2, H, W)
#             else:
#                 logits = model(image.unsqueeze(0).to(device))          # 原来的写法
#             p = torch.softmax(logits, dim=1)[0, 0]
#             probs.append(p.cpu().numpy())
#     return np.stack(probs)

def predict_probs(model_path, ds, padding=1, tta=False):
    model = UNet(padding=padding).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    probs = []
    with torch.no_grad():
        for i in range(len(ds)):
            image, _, _ = ds[i]
            if tta:
                p = predict_tta(model, image, padding)
            else:
                p = predict_one(model, image, padding)
            probs.append(p.numpy())
    return np.stack(probs)

def f_score(pred, gt):
    """pred、gt 都是 bool 数组，True = 细胞内部"""
    tp = (pred & gt).sum()
    fp = (pred & ~gt).sum()
    fn = (~pred & gt).sum()
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def pixel_error(probs, gt):
    """官方定义：1 − 在阈值 0.0, 0.1, …, 1.0 中取到的最大 F-score"""
    best = 0.0
    for t in np.arange(0.0, 1.01, 0.1):
        pred = probs > t
        best = max(best, f_score(pred, gt))
    return 1 - best

def rand_counts(pred_cell, gt_cell):
    gt_seg, _ = ndimage.label(gt_cell)
    pred_seg, _ = ndimage.label(pred_cell)
    mask = gt_seg > 0
    g = gt_seg[mask].astype(np.int64)
    p = pred_seg[mask].astype(np.int64)

    fg = p > 0                    # 预测为细胞的像素
    n0 = (~fg).sum()              # 预测为膜的像素个数，每个算一个单像素片段
    _, n_ij = np.unique(g[fg] * (p.max() + 1) + p[fg], return_counts=True)
    _, a_i = np.unique(g, return_counts=True)
    _, b_j = np.unique(p[fg], return_counts=True)
    return (n_ij ** 2).sum() + n0, (a_i ** 2).sum(), (b_j ** 2).sum() + n0


def rand_error(probs, gt, verbose=False):
    best = 0.0
    for t in np.arange(0.0, 1.01, 0.1):
        S_both = S_gt = S_pred = 0
        for k in range(len(probs)):
            s_both, s_gt, s_pred = rand_counts(probs[k] > t, gt[k])
            S_both += s_both
            S_gt += s_gt
            S_pred += s_pred
        precision = S_both / S_pred
        recall = S_both / S_gt
        f = 2 * precision * recall / (precision + recall)
        if verbose:
            print(f"  t={t:.1f}  P={precision:.4f}  R={recall:.4f}  F={f:.4f}")
        best = max(best, f)
    return 1 - best

def predict_tta(model, image, padding):
    """8 种旋转/翻转分别预测，还原后取平均"""
    probs = []
    for k in range(4):
        for flip in [False, True]:
            x = torch.rot90(image, k, dims=(1, 2))          # image 是 (1, H, W)，在哪两个维度上旋转？
            if flip:
                x = torch.flip(x, dims=[2])                 # 左右翻转

            p = predict_one(model, x, padding)              # (H, W)

            # 逆变换：顺序和上面相反
            if flip:
                p = torch.flip(p, dims=[1])               # p 是 (H, W)，左右翻转是第几维？
            p = torch.rot90(p, -k, dims=(0, 1))            # 转回来：转 k 次的逆是什么？
            probs.append(p)
    return torch.stack(probs).mean(dim=0)

def rand_per_slice(probs, gt, t):
    """在阈值 t 下，每张图单独的 Rand F 值"""
    fs = []
    for k in range(len(probs)):
        s_both, s_gt, s_pred = rand_counts(probs[k] > t, gt[k])
        p = s_both / s_pred
        r = s_both / s_gt
        fs.append(2 * p * r / (p + r))
    return np.array(fs)

if __name__ == "__main__":
    test_ds = ISBIDataset("ISBI-2012-challenge/test-volume.tif",
                          "ISBI-2012-challenge/test-labels.tif")

    img = test_ds[0][0]                                  # (1, 512, 512)
    for k in range(4):
        for flip in [False, True]:
            x = torch.rot90(img, k, dims=(1, 2))
            if flip:
                x = torch.flip(x, dims=[2])
            p = x[0]                                     # 假装"模型"原样输出 → (H, W)
            if flip:
                p = torch.flip(p, dims=[1])
            p = torch.rot90(p, -k, dims=(0, 1))
            print(k, flip, torch.equal(p, img[0]))       # 8 行都应该是 True
    # for k in range(4):
    #     for flip in [False, True]:
    #         ...
    #         print(k, flip, torch.equal(p, img[0]))

    gt = np.stack([test_ds[i][1].numpy() == 0 for i in range(len(test_ds))])   # True = 细胞内部
    subset = list(range(1, 30, 3))           # 官方 "33%"：第 2、5、…、29 张（从 0 开始数就是 1、4、…、28）
    fake_probs = gt.astype(np.float32)
    print("自我检查（应该都是 0）: pixel error =", pixel_error(fake_probs, gt),
          " Rand error =", rand_error(fake_probs, gt))

    models = {
        "E8": ("best_model_bs1_lr1e-3.pth", 1),
        "E11": ("best_model_weighted.pth", 1),
        "E12 (val_loss)": ("best_model_weighted_dropout.pth", 1),
        "E12 (F1)": ("best_model_weighted_dropout_f1.pth", 1),
        "E13 (val_loss)": ("best_model_nopad.pth", 0),
        "E13 (F1)": ("best_model_nopad_f1.pth", 0),
    }

    eval_names = ["E8", "E12 (F1)", "E13 (F1)"]

    # print(f"{'模型':<16}{'TTA':>5}{'pixel(30)':>12}{'pixel(10)':>12}{'Rand(30)':>12}{'Rand(10)':>12}")
    # for name in eval_names:
    #     path, padding = models[name]
    #     for tta in [False, True]:
    #         probs = predict_probs(path, test_ds, padding=padding, tta=tta)
    #         e_all = pixel_error(probs, gt)
    #         e_sub = pixel_error(probs[subset], gt[subset])
    #         r_all = rand_error(probs, gt)
    #         r_sub = rand_error(probs[subset], gt[subset])
    #         print(f"{name:<16}{str(tta):>5}{e_all:>12.4f}{e_sub:>12.4f}{r_all:>12.4f}{r_sub:>12.4f}")
    # print("论文 U-Net: pixel error 0.0611, Rand error 0.0382")

    # seed_models = {
    #     "seed 42": "best_model_nopad_f1.pth",
    #     "seed 1": "best_model_nopad_f1_seed1.pth",
    #     "seed 2": "best_model_nopad_f1_seed2.pth",
    # }
    #
    # for tta in [False, True]:
    #     print(f"\nE13 (F1)，TTA = {tta}")
    #     print(f"{'种子':<10}{'pixel(30)':>12}{'Rand(30)':>12}")
    #     results = []
    #     for name, path in seed_models.items():
    #         probs = predict_probs(path, test_ds, padding=0, tta=tta)
    #         e = pixel_error(probs, gt)
    #         r = rand_error(probs, gt)
    #         results.append([e, r])
    #         print(f"{name:<10}{e:>12.4f}{r:>12.4f}")
    #
    #     results = np.array(results)                  # (3, 2)：3 个种子 × 2 个指标
    #     mean = results.mean(axis=0)                # 对哪一维求平均？
    #     std = results.std(axis=0, ddof=1)
    #     print(f"{'均值':<10}{mean[0]:>12.4f}{mean[1]:>12.4f}")
    #     print(f"{'标准差':<10}{std[0]:>12.4f}{std[1]:>12.4f}")
    probs = predict_probs("best_model_nopad_f1.pth", test_ds, padding=0)
    np.save("probs_e13_s42.npy", probs)          # 存下来，下次直接 np.load，不用再预测

    fs = rand_per_slice(probs, gt, t=0.6)
    order = np.argsort(fs)                       # 从差到好排序
    for k in order:
        print(f"第 {k:2d} 张  Rand F = {fs[k]:.4f}  Rand error = {1 - fs[k]:.4f}")

    # for name in ["E12 (F1)", "E13 (F1)"]:
    #     print(f"\n{name} 各阈值的 Rand precision / recall（30 张）:")
    #     path, padding = models[name]
    #     probs = predict_probs(path, test_ds, padding=padding)
    #     rand_error(probs, gt, verbose=True)
