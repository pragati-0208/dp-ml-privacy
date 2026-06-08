"""
utils.py
--------
Data loading, metric tracking, and logging utilities.
"""

import os
import json
import time
import logging
from typing import Tuple, Dict, List, Optional

import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split

# ── Logging ──────────────────────────────────────────────────────────────────

def setup_logger(name: str = "dp_ml", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s  %(message)s", datefmt="%H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger

logger = setup_logger()


# ── Data Loading ─────────────────────────────────────────────────────────────

def get_cifar10_transforms(augment: bool = True):
    """
    Returns train and test transforms for CIFAR-10.

    With DP we typically want lighter augmentation so the training signal
    is cleaner (augmentation adds variance that competes with DP noise),
    but standard augmentation for non-DP baselines.
    """
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std =[0.2023, 0.1994, 0.2010],
    )
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([transforms.ToTensor(), normalize])

    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    return train_transform, test_transform


def get_data_loaders(
    data_dir: str = "./data",
    batch_size: int = 256,
    num_workers: int = 2,
    subset_fraction: float = 1.0,
    augment: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Returns (train_loader, test_loader) for CIFAR-10.

    Args:
        subset_fraction: Use only this fraction of training data.
                         Useful for quick experiments during development.
        augment: Whether to apply random crop / flip on training set.
    """
    train_transform, test_transform = get_cifar10_transforms(augment=augment)

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    if subset_fraction < 1.0:
        rng = torch.Generator().manual_seed(seed)
        n = int(len(train_dataset) * subset_fraction)
        train_dataset, _ = random_split(train_dataset, [n, len(train_dataset) - n], generator=rng)
        logger.info(f"Using subset of training data: {n} samples ({subset_fraction*100:.0f}%)")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,   # Required by Opacus — partial batches break per-sample clipping
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(f"Train batches: {len(train_loader)} | Test samples: {len(test_dataset)}")
    return train_loader, test_loader


# ── Metrics ───────────────────────────────────────────────────────────────────

class MetricTracker:
    """
    Lightweight metric tracker.
    Records train_loss, train_acc, test_loss, test_acc per epoch.
    Supports JSON serialization for saving/loading experiment results.
    """

    def __init__(self):
        self.history: Dict[str, List] = {
            "epoch": [], "train_loss": [], "train_acc": [],
            "test_loss": [], "test_acc": [], "epsilon": [], "elapsed_s": [],
        }
        self._start_time = time.time()

    def update(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        test_loss: float,
        test_acc: float,
        epsilon: Optional[float] = None,
    ):
        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(round(train_loss, 4))
        self.history["train_acc"].append(round(train_acc, 4))
        self.history["test_loss"].append(round(test_loss, 4))
        self.history["test_acc"].append(round(test_acc, 4))
        self.history["epsilon"].append(epsilon)
        self.history["elapsed_s"].append(round(time.time() - self._start_time, 1))

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Metrics saved → {path}")

    @classmethod
    def load(cls, path: str) -> "MetricTracker":
        tracker = cls()
        with open(path) as f:
            tracker.history = json.load(f)
        return tracker

    def best_test_acc(self) -> float:
        accs = self.history["test_acc"]
        return max(accs) if accs else 0.0


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Returns (avg_loss, accuracy) over the full loader."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


# ── Misc ──────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        logger.info("Using GPU: " + torch.cuda.get_device_name(0))
        return torch.device("cuda")
    logger.info("Using CPU (no GPU found)")
    return torch.device("cpu")


def save_checkpoint(model: torch.nn.Module, path: str, metadata: dict = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"state_dict": model.state_dict(), "metadata": metadata or {}}
    torch.save(payload, path)
    logger.info(f"Checkpoint saved → {path}")
