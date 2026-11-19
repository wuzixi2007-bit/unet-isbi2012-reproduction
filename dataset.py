import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn.functional as F
from weights import compute_weight_map
from tile import IN, OUT, MARGIN

def load_tif_stack(path):
    img = Image.open(path)  # ① 打开文件
    frames = []  # 用来存放每一页
    for i in range(img.n_frames):  # ② 一共有多少页？
        img.seek(i)  # ③ 切换到第 i 页
        frames.append(np.array(img))
    return np.stack(frames)  # ④ 把列表里的 30 个数组叠成一个

# 弹性形变
def elastic_deform(image, label,weight, sigma=10.0, grid_size=3):
    _, H, W = image.shape

    # ① 在 3×3 网格上随机生成位移（单位：像素），2 个通道分别是 dx 和 dy
    disp = torch.randn(2, grid_size, grid_size) * sigma

    # ② 双三次插值，放大成每个像素的位移 → (2, H, W)
    disp = F.interpolate(disp.unsqueeze(0), size=(H, W),
                         mode="bicubic", align_corners=True)[0]

    # ③ 像素位移 → 归一化坐标下的位移
    dx = disp[0] * 2 / (W - 1)
    dy = disp[1] * 2 / (H - 1)

    # ④ 构造取值位置表 grid：原始坐标 + 位移
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, H),
                            torch.linspace(-1, 1, W), indexing="ij")
    grid = torch.stack([xs+dx, ys+dy], dim=-1).unsqueeze(0)    # (1, H, W, 2)，顺序是 (x, y)

    # ⑤ 按照 grid 从原图取值
    image = F.grid_sample(image.unsqueeze(0), grid, mode="bilinear",
                          padding_mode="reflection", align_corners=True)[0]
    label = F.grid_sample(label[None, None].float(), grid, mode="nearest",
                          padding_mode="reflection", align_corners=True)[0, 0].long()
    weight = F.grid_sample(weight[None, None], grid, mode="bilinear",
                           padding_mode="reflection", align_corners=True)[0, 0]

    return image, label,weight

class ISBIDataset(Dataset):
    def __init__(self, image_path, label_path, augment=False, elastic=False, wc=None,crop=False):
        self.images = load_tif_stack(image_path)
        self.labels = load_tif_stack(label_path)
        self.augment = augment
        self.elastic = elastic
        self.crop = crop

        # 提前算好所有权重图；不传 wc 时不用权重图
        self.weights = None
        if wc is not None:
            labels01 = (self.labels == 0).astype(np.int64)      # 细胞膜 = 1
            print("正在计算权重图……")
            self.weights = np.stack([compute_weight_map(l, wc) for l in labels01])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.weights is not None:
            weight = self.weights[idx]
        else:
            weight = np.ones(label.shape, dtype=np.float32)     # 没有权重图时，权重全是 1

        if self.augment:
            k = torch.randint(0, 4, (1,)).item()
            image = np.rot90(image, k)
            label = np.rot90(label, k)
            weight = np.rot90(weight, k)                                        # 和标注做同样的旋转

            if torch.rand(1).item() < 0.5:
                image = np.fliplr(image)
                label = np.fliplr(label)
                weight = np.fliplr(weight)

            image = image.copy()
            label = label.copy()
            weight = weight.copy()

        image = image.astype(np.float32) / 255.0
        image = image[np.newaxis, :, :]
        label = (label == 0).astype(np.int64)

        image = torch.from_numpy(image)
        label = torch.from_numpy(label).long()
        weight = torch.from_numpy(weight)

        if self.augment and self.elastic:
            image, label, weight = elastic_deform(image, label, weight)

        if self.crop:
            _, H, W = image.shape
            y = torch.randint(0, H - OUT + 1, (1,)).item()     # 输出块左上角的行
            x = torch.randint(0, W - OUT + 1, (1,)).item()             # 列
            self.last_yx = (y, x)                              # 记下来，只是为了下面的自我检查

            padded = F.pad(image[None], (MARGIN, MARGIN, MARGIN, MARGIN), mode="reflect")[0]
            image = padded[:, y:y + IN, x:x + IN]
            label = label[y:y + OUT, x:x + OUT]
            weight = weight[y:y + OUT, x:x + OUT]

        return image, label, weight


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    path_img = "ISBI-2012-challenge/train-volume.tif"
    path_lbl = "ISBI-2012-challenge/train-labels.tif"

    plain = ISBIDataset(path_img, path_lbl)                 # 不裁剪
    cropped = ISBIDataset(path_img, path_lbl, crop=True)    # 只裁剪，不增强
    img_c, lbl_c, w_c = cropped[0]
    img_p, lbl_p, w_p = plain[0]
    y, x = cropped.last_yx
    print("裁剪后形状:", img_c.shape, lbl_c.shape, w_c.shape)   # 应该是 (1,572,572) (388,388) (388,388)
    print("位置:", (y, x))
    print("标注对得上吗:", torch.equal(lbl_c, lbl_p[y:y + OUT, x:x + OUT]))
    print("图片中心对得上吗:",
          torch.equal(img_c[:, MARGIN:MARGIN + OUT, MARGIN:MARGIN + OUT],
                      img_p[:, y:y + OUT, x:x + OUT]))


    ds_plain = ISBIDataset(path_img, path_lbl)                          # 不增强
    ds_elastic = ISBIDataset(path_img, path_lbl, augment=True, elastic=True,
                             wc={0: 0.641, 1: 2.277})

    fig, axes = plt.subplots(3, 4, figsize=(16, 8))
    img, lbl, w = ds_plain[0]
    axes[0, 0].imshow(img[0], cmap="gray")
    axes[0, 0].set_title("Original")
    axes[1, 0].imshow(img[0], cmap="gray")
    axes[1, 0].imshow(lbl, cmap="Reds", alpha=0.4)                      # 标注叠在图片上
    axes[1, 0].set_title("Original + Label")
    axes[2, 0].imshow(w, cmap="jet")
    axes[2, 0].set_title("Original weight")

    for i in range(1, 4):
        img, lbl, w = ds_elastic[0]
        axes[0, i].imshow(img[0], cmap="gray")
        axes[0, i].set_title(f"Elastic #{i}")
        axes[1, i].imshow(img[0], cmap="gray")
        axes[1, i].imshow(lbl, cmap="Reds", alpha=0.4)
        axes[1, i].set_title(f"Elastic #{i} + Label")
        axes[2, i].imshow(w, cmap="jet")
        axes[2, i].set_title(f"Elastic #{i} weight")

    img, lbl, w = ds_elastic[0]
    print("图片:", img.shape, img.dtype, "标注:", lbl.shape, lbl.dtype, torch.unique(lbl))
    plt.show()

