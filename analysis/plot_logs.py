"""Plot the retained E8/E13 repeated-run logs."""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "results" / "logs"


def read_log(path: Path):
    epochs, train_loss, val_loss, val_acc = [], [], [], []
    pattern = re.compile(
        r"epoch (\d+)/\d+\s+train_loss = ([\d.]+)\s+"
        r"val_loss = ([\d.]+)\s+val_acc = ([\d.]+)"
    )
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if match:
            epochs.append(int(match.group(1)))
            train_loss.append(float(match.group(2)))
            val_loss.append(float(match.group(3)))
            val_acc.append(float(match.group(4)))
    return epochs, train_loss, val_loss, val_acc


fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = {42: "tab:blue", 1: "tab:orange", 2: "tab:green"}
for preset, style in (("e8", "-"), ("e13", "--")):
    for seed in (42, 1, 2):
        epochs, _, val_loss, val_acc = read_log(LOG_DIR / f"train_log_{preset}_seed{seed}.txt")
        label = f"{preset.upper()} seed {seed}"
        axes[0].plot(epochs, val_loss, style, color=colors[seed], label=label)
        axes[1].plot(epochs, val_acc, style, color=colors[seed], label=label)

axes[0].set(xlabel="epoch", ylabel="validation loss", title="Validation loss")
axes[1].set(xlabel="epoch", ylabel="validation accuracy", title="Validation accuracy")
for axis in axes:
    axis.legend(fontsize=8)
    axis.grid(alpha=0.2)

fig.tight_layout()
output = ROOT / "results" / "training_curves.png"
fig.savefig(output, dpi=140)
print(output)
