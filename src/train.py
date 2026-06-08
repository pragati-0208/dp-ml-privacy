"""
train.py
--------
Standard training loop and DP-SGD training loop.

Key idea: the DP training loop is nearly identical to the standard one.
The magic happens inside the Opacus privacy engine, which hooks into
PyTorch's autograd to do per-sample gradient clipping before the
optimizer step. We never touch the clipping logic ourselves.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import evaluate, MetricTracker, logger, save_checkpoint


# ── Standard (non-private) training ──────────────────────────────────────────

def train_epoch_standard(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """One epoch of standard (non-DP) training. Returns (loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="  train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def train_standard(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int,
    lr: float,
    device: torch.device,
    save_dir: str = "./checkpoints",
) -> MetricTracker:
    """Full standard training run. Returns MetricTracker with history."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    tracker = MetricTracker()

    logger.info(f"Starting standard training for {num_epochs} epochs")

    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = train_epoch_standard(model, train_loader, optimizer, criterion, device)
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        tracker.update(epoch, tr_loss, tr_acc, te_loss, te_acc, epsilon=None)
        logger.info(
            f"Epoch {epoch:3d}/{num_epochs} | "
            f"train loss={tr_loss:.4f} acc={tr_acc:.3f} | "
            f"test  loss={te_loss:.4f} acc={te_acc:.3f}"
        )

    save_checkpoint(model, f"{save_dir}/standard_final.pt", {"epochs": num_epochs})
    return tracker


# ── DP-SGD training ───────────────────────────────────────────────────────────

def train_epoch_dp(
    model: nn.Module,
    loader: DataLoader,
    optimizer,               # This is a DPOptimizer from Opacus
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    One epoch of DP training.

    The key difference from standard training:
    - optimizer.zero_grad() must be called with set_to_none=False in Opacus ≥1.4
    - Everything else looks the same — Opacus hooks handle clipping + noise injection

    Opacus enforces per-sample gradients, so the model must not have in-place
    operations or BatchNorm (we use GroupNorm in our architecture).
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="  train(DP)", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def train_dp(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int,
    lr: float,
    device: torch.device,
    target_epsilon: float,
    target_delta: float,
    max_grad_norm: float,
    save_dir: str = "./checkpoints",
) -> Tuple[MetricTracker, float]:
    """
    Full DP-SGD training run using Opacus.

    Args:
        target_epsilon: Privacy budget. Lower = more private = more noise.
        target_delta:   Probability of epsilon being exceeded. Typically 1/N.
        max_grad_norm:  L2 clipping threshold for per-sample gradients.
                        Smaller = more clipping = less information leaked per step.

    Returns:
        (MetricTracker, actual_epsilon_spent)

    How Opacus works under the hood:
        1. PrivacyEngine.make_private() wraps the model, optimizer, and loader.
        2. On each backward pass, Opacus computes per-sample gradients (not averaged).
        3. Each per-sample gradient is clipped to max_grad_norm.
        4. Gaussian noise N(0, sigma^2 * max_grad_norm^2) is added to the sum.
        5. The accountant tracks the cumulative privacy cost (epsilon, delta).
    """
    # Import here so the file is importable even if opacus isn't installed
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator

    # Opacus requires the model to pass its validator (no BatchNorm, no in-place ops)
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        logger.warning(f"Model validation warnings: {errors}")
        model = ModuleValidator.fix(model)
        logger.info("Model auto-fixed by Opacus validator.")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    privacy_engine = PrivacyEngine()

    # make_private_with_epsilon automatically computes the noise multiplier (sigma)
    # needed to hit target_epsilon after num_epochs of training.
    # This is the recommended way — you specify what epsilon you WANT,
    # Opacus figures out what sigma achieves it.
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=train_loader.dataset.__class__.__name__ and model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=num_epochs,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        max_grad_norm=max_grad_norm,
    )

    tracker = MetricTracker()
    logger.info(
        f"Starting DP training | ε={target_epsilon} δ={target_delta} "
        f"C={max_grad_norm} | {num_epochs} epochs"
    )

    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = train_epoch_dp(model, train_loader, optimizer, criterion, device)
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        epsilon_spent = privacy_engine.get_epsilon(target_delta)

        tracker.update(epoch, tr_loss, tr_acc, te_loss, te_acc, epsilon=epsilon_spent)
        logger.info(
            f"Epoch {epoch:3d}/{num_epochs} | "
            f"train acc={tr_acc:.3f} | test acc={te_acc:.3f} | "
            f"ε spent={epsilon_spent:.3f}"
        )

    final_epsilon = privacy_engine.get_epsilon(target_delta)
    save_checkpoint(
        model,
        f"{save_dir}/dp_eps{target_epsilon:.1f}_final.pt",
        {"target_epsilon": target_epsilon, "final_epsilon": final_epsilon},
    )
    return tracker, final_epsilon
