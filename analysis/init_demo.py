import torch
import torch.nn as nn

torch.manual_seed(0)


def make_net(use_he):
    layers = []
    for _ in range(15):                       # 15 层"卷积 + ReLU"，64 通道
        conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        if use_he:
            nn.init.kaiming_normal_(conv.weight, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(conv.bias)
        layers += [conv, nn.ReLU()]
    return nn.Sequential(*layers)


x = torch.randn(1, 64, 64, 64)                # 方差为 1 的输入

for use_he in [False, True]:
    net = make_net(use_he)
    name = "He 初始化" if use_he else "默认初始化"
    print(f"\n===== {name} =====")
    h = x
    with torch.no_grad():
        for i, layer in enumerate(net):
            h = layer(h)
            if isinstance(layer, nn.ReLU):
                print(f"第 {i // 2 + 1:2d} 层输出的标准差: {h.std().item():.6f}")