import torch
import torch.nn.functional as F

IN, OUT = 572, 388
MARGIN = (IN - OUT) // 2          # 每边需要镜像填充多少像素？算一下应该是 92


def predict_tiled(model, image, starts=(0, 128)):
    """image: (1, H, W)，H = W = 512  →  logits: (2, H, W)"""
    device = next(model.parameters()).device
    _, H, W = image.shape
    extra = starts[-1] + OUT - H                  # 最后一块超出原图多少（516 - 512 = 4）

    # 镜像填充：左、右、上、下各填多少？右边和下边还要多填 extra
    padded = F.pad(image[None], (MARGIN, MARGIN + extra, MARGIN, MARGIN + extra),
                   mode="reflect")[0]             # (1, H + 2*MARGIN + extra, ...)

    out = torch.zeros(2, H + extra, W + extra)
    with torch.no_grad():
        for y in starts:
            for x in starts:
                tile = padded[:, y:y + IN, x:x + IN]                 # 输入块多大？
                logits = model(tile[None].to(device))[0].cpu()         # (2, 388, 388)
                out[:, y:y + OUT, x:x + OUT] = logits                  # 输出块多大？
    return out[:, :H, :W]                         # 裁掉多出来的部分

if __name__ == "__main__":
    from dataset import ISBIDataset
    from model import UNet

    torch.manual_seed(0)
    model = UNet(padding=0).eval()                # 没训练过的模型就够了
    image = ISBIDataset("ISBI-2012-challenge/train-volume.tif",
                        "ISBI-2012-challenge/train-labels.tif")[0][0]

    print("输出形状:", predict_tiled(model, image).shape)   # 应该是 (2, 512, 512)

    padded = F.pad(image[None], (MARGIN, MARGIN + 4, MARGIN, MARGIN + 4), mode="reflect")[0]
    for shift in [128, 124]:
        with torch.no_grad():
            a = model(padded[None, :, 0:IN, 0:IN])                  # 左边的块
            b = model(padded[None, :, 0:IN, shift:shift + IN])      # 右边的块
        overlap = OUT - shift                                       # 重叠多少列
        diff = (a[..., shift:] - b[..., :overlap]).abs().max().item()
        print(f"起点 {shift}：重叠区域最大差异 = {diff:.2e}")