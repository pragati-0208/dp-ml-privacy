"""
experiments/run_baseline.py
----------------------------
Train a standard (non-private) CNN on CIFAR-10.

This gives us the performance ceiling — the best possible accuracy
without any privacy constraints. Every DP experiment is measured
as accuracy degradation relative to this baseline.

Usage:
    python experiments/run_baseline.py
    python experiments/run_baseline.py --epochs 30 --lr 1e-3
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import get_model
from src.utils import get_data_loaders, get_device, set_seed, logger
from src.train import train_standard


def parse_args():
    p = argparse.ArgumentParser(description="Train baseline (non-DP) CNN on CIFAR-10")
    p.add_argument("--epochs",      type=int,   default=20,    help="Training epochs")
    p.add_argument("--lr",          type=float, default=1e-3,  help="Learning rate")
    p.add_argument("--batch-size",  type=int,   default=256,   help="Batch size")
    p.add_argument("--data-dir",    type=str,   default="./data")
    p.add_argument("--results-dir", type=str,   default="./results")
    p.add_argument("--ckpt-dir",    type=str,   default="./checkpoints")
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    logger.info("=" * 60)
    logger.info("  Baseline Training (No Differential Privacy)")
    logger.info("=" * 60)

    train_loader, test_loader = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )

    model = get_model()
    logger.info(f"Model parameters: {model.count_parameters():,}")

    tracker = train_standard(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        device=device,
        save_dir=args.ckpt_dir,
    )

    os.makedirs(args.results_dir, exist_ok=True)
    tracker.save(os.path.join(args.results_dir, "baseline_metrics.json"))

    logger.info(f"\nBest test accuracy: {tracker.best_test_acc():.4f}")
    logger.info("Baseline training complete.")


if __name__ == "__main__":
    main()
