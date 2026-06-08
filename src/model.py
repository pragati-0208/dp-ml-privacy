"""
model.py
--------
CNN architecture for CIFAR-10 classification.

Design choices:
- Small enough to train fast on CPU/single GPU (important for a sweep over many epsilon values)
- BatchNorm replaced with GroupNorm — Opacus requires this because BatchNorm
  computes statistics across the batch, which leaks per-sample information and
  is incompatible with per-sample gradient clipping.
- Modular: easy to swap in a deeper model later
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv -> GroupNorm -> ReLU block. GroupNorm works with Opacus; BatchNorm does not."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        # num_groups=out_channels makes this equivalent to InstanceNorm per channel
        # num_groups=1 is LayerNorm. We use min(8, out_channels) as a balanced default.
        num_groups = min(8, out_channels)
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class CIFAR10CNN(nn.Module):
    """
    Small CNN for CIFAR-10.

    Architecture:
        Block1: 3  -> 32 channels, 3x3 conv, MaxPool
        Block2: 32 -> 64 channels, 3x3 conv, MaxPool
        Block3: 64 -> 128 channels, 3x3 conv, MaxPool (now 4x4 spatial)
        FC1:    128*4*4 -> 256
        FC2:    256 -> 10 (logits)

    ~500K parameters — fast to train, good enough to show the DP tradeoff clearly.
    """

    def __init__(self, num_classes: int = 10, dropout_p: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            nn.MaxPool2d(2, 2),          # 32x32 -> 16x16
            ConvBlock(32, 64),
            nn.MaxPool2d(2, 2),          # 16x16 -> 8x8
            ConvBlock(64, 128),
            nn.MaxPool2d(2, 2),          # 8x8 -> 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_model(num_classes: int = 10) -> CIFAR10CNN:
    """Factory function — always returns a fresh model on CPU initially."""
    model = CIFAR10CNN(num_classes=num_classes)
    return model


if __name__ == "__main__":
    model = get_model()
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    print(f"Output shape : {out.shape}")          # (4, 10)
    print(f"Parameters   : {model.count_parameters():,}")
