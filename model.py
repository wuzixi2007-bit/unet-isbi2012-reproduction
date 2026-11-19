import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(3×3 卷积 → ReLU) × 2"""

    def __init__(self, in_channels, out_channels,padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels,out_channels, kernel_size=3, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class Down(nn.Module):
    """2×2 最大池化 → DoubleConv"""

    def __init__(self, in_channels, out_channels, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels, padding=padding),
        )

    def forward(self, x):
        return self.block(x)


def center_crop(skip, target):
    """把 skip 中心裁剪到和 target 一样的高和宽"""
    _, _, h, w = target.shape          # 目标的高和宽
    _, _, H, W = skip.shape            # skip 的高和宽
    top = (H - h) // 2                   # 上边要裁掉几行？
    left = (W - w) // 2                  # 左边要裁掉几列？
    return skip[:, :, top:top + h, left:left + w]


class Up(nn.Module):
    """2×2 转置卷积 → 裁剪并拼接 skip → DoubleConv"""

    def __init__(self, in_channels, out_channels, padding=1):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, padding=padding)

    def forward(self, x, skip):
        x = self.up(x)
        skip = center_crop(skip, x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, padding=1, dropout=0.0):
        super().__init__()
        # 编码器
        self.inc = DoubleConv(in_channels, 64, padding=padding)   # 第 1 层，没有池化
        self.down1 = Down(64, 128, padding=padding)
        self.down2 = Down(128, 256, padding=padding)
        self.down3 = Down(256, 512, padding=padding)
        self.down4 = Down(512, 1024, padding=padding)              # 最底层
        self.dropout = nn.Dropout(p=dropout)

        # 解码器
        self.up1 = Up(1024, 512, padding=padding)
        self.up2 = Up(512, 256, padding=padding)
        self.up3 = Up(256, 128, padding=padding)
        self.up4 = Up(128, 64, padding=padding)

        # 输出层：1×1 卷积
        self.outc = nn.Conv2d(64, num_classes, kernel_size=1)

        #He 初始化
        self._init_weights()

    def _init_weights(self):
        """论文第 9 题：He 初始化，偏置为 0"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.dropout(self.down3(x3))
        x5 = self.dropout(self.down4(x4))

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)

if __name__ == "__main__":
    # 简化版
    model = UNet()
    x = torch.randn(1, 1, 512, 512)
    print("简化版:", x.shape, "→", model(x).shape)

    # 论文版
    model_paper = UNet(padding=0)
    x = torch.randn(1, 1, 572, 572)
    print("论文版:", x.shape, "→", model_paper(x).shape)

    # 数一数卷积层
    n_conv = sum(1 for m in model.modules()
                 if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)))
    print("卷积层数:", n_conv)

    # 数一数参数量
    n_params = sum(p.numel() for p in model.parameters())
    print("参数量:", n_params)

    print("5 类:", UNet(num_classes=5)(torch.randn(1, 1, 512, 512)).shape)