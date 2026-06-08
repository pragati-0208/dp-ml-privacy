"""
experiments/run_attack.py
--------------------------
Run Membership Inference Attack on both:
    1. A standard (non-private) model
    2. A DP-trained model (epsilon = 2.0)

This produces the key result that motivates DP:
    The standard model is VULNERABLE to MIA.
    The DP model is RESISTANT.

Usage:
    python experiments/run_attack.py
    python experiments/run_attack.py --epsilon 5.0 --epochs 15
"""

import argparse
import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np

from src.model import get_model
from src.utils import get_device, set_seed, logger, evaluate, get_cifar10_transforms
from src.attacks import threshold_attack, evaluate_attack_vs_dp
from src.train import train_standard
from src.privacy_engine import DPConfig, make_private


def get_attack_loaders(data_dir: str, batch_size: int = 128):
    """
    Create member and non-member loaders for the attack.

    Split:
        - First 10,000 train samples  → "members" (model was trained on these)
        - First 10,000 test samples   → "non-members" (model never saw these)

    Using equal-sized splits ensures no class imbalance in the attack classifier.
    """
    train_transform, test_transform = get_cifar10_transforms(augment=False)

    full_train = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    full_test = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    n_attack = 2000  # 2K each for fast evaluation

    member_loader = DataLoader(
        Subset(full_train, list(range(n_attack))),
        batch_size=batch_size, shuffle=False
    )
    nonmember_loader = DataLoader(
        Subset(full_test, list(range(n_attack))),
        batch_size=batch_size, shuffle=False
    )

    # Separate splits for shadow model training
    shadow_member_loader = DataLoader(
        Subset(full_train, list(range(n_attack, 2 * n_attack))),
        batch_size=batch_size, shuffle=False
    )
    shadow_nonmember_loader = DataLoader(
        Subset(full_test, list(range(n_attack, 2 * n_attack))),
        batch_size=batch_size, shuffle=False
    )

    # Full loaders for model training
    full_train_loader = DataLoader(
        full_train, batch_size=256, shuffle=True, drop_last=True
    )
    full_test_loader = DataLoader(
        full_test, batch_size=512, shuffle=False
    )

    return (
        member_loader, nonmember_loader,
        shadow_member_loader, shadow_nonmember_loader,
        full_train_loader, full_test_loader
    )


def train_model_standard(train_loader, test_loader, device, epochs, lr):
    """Train a non-private model and return it."""
    set_seed(42)
    model = get_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    logger.info(f"Training standard model for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        if epoch % 5 == 0:
            _, acc = evaluate(model, test_loader, criterion, device)
            logger.info(f"  Epoch {epoch}/{epochs} | test acc={acc:.3f}")

    return model


def train_model_dp(train_loader, test_loader, device, epochs, lr, epsilon, delta=1e-5):
    """Train a DP model and return it."""
    set_seed(42)
    model = get_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    config = DPConfig(target_epsilon=epsilon, target_delta=delta,
                      max_grad_norm=1.0, num_epochs=epochs)

    model, optimizer, dp_loader, privacy_engine = make_private(
        model, optimizer, train_loader, config
    )

    logger.info(f"Training DP model (ε={epsilon}) for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        for images, labels in dp_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        if epoch % 5 == 0:
            _, acc = evaluate(model, test_loader, criterion, device)
            eps_spent = privacy_engine.get_epsilon(delta)
            logger.info(f"  Epoch {epoch}/{epochs} | test acc={acc:.3f} | ε={eps_spent:.3f}")

    return model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epsilon",     type=float, default=2.0)
    p.add_argument("--epochs",      type=int,   default=20)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--data-dir",    type=str,   default="./data")
    p.add_argument("--results-dir", type=str,   default="./results")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()

    (member_loader, nonmember_loader,
     shadow_member_loader, shadow_nonmember_loader,
     train_loader, test_loader) = get_attack_loaders(args.data_dir)

    # ── Train standard model ──────────────────────────────────────────────────
    logger.info("\n" + "=" * 55)
    logger.info("  Step 1: Training Standard (Non-Private) Model")
    logger.info("=" * 55)
    standard_model = train_model_standard(train_loader, test_loader, device, args.epochs, args.lr)

    # ── Run attack on standard model ──────────────────────────────────────────
    logger.info("\n  Running MIA on standard model...")
    results_no_dp = threshold_attack(standard_model, member_loader, nonmember_loader, device)
    logger.info(f"  Attack AUC (no DP): {results_no_dp['auc']:.4f}")

    # ── Train DP model ────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 55)
    logger.info(f"  Step 2: Training DP Model (ε={args.epsilon})")
    logger.info("=" * 55)
    dp_model = train_model_dp(
        train_loader, test_loader, device, args.epochs, args.lr, args.epsilon
    )

    # ── Run attack on DP model ────────────────────────────────────────────────
    logger.info("\n  Running MIA on DP model...")
    results_dp = threshold_attack(dp_model, member_loader, nonmember_loader, device)
    logger.info(f"  Attack AUC (DP, ε={args.epsilon}): {results_dp['auc']:.4f}")

    # ── Print comparison ──────────────────────────────────────────────────────
    print("\n" + evaluate_attack_vs_dp(results_no_dp, results_dp))

    # ── Save results ──────────────────────────────────────────────────────────
    os.makedirs(args.results_dir, exist_ok=True)
    output = {
        "no_dp": results_no_dp,
        "dp": results_dp,
        "epsilon_used": args.epsilon,
        "note": "AUC near 0.5 = attack fails = good privacy",
    }
    path = os.path.join(args.results_dir, "attack_results.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Attack results saved → {path}")


if __name__ == "__main__":
    main()
