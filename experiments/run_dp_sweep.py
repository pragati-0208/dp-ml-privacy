"""
experiments/run_dp_sweep.py
----------------------------
Sweep over different epsilon values and record accuracy.

This is the experiment that produces the main result:
    Privacy-Accuracy Tradeoff Curve

    epsilon = 0.5  → very strong privacy, low accuracy
    epsilon = 1.0  → strong privacy, moderate accuracy
    epsilon = 2.0  → moderate privacy
    epsilon = 5.0  → weak privacy
    epsilon = 10.0 → very weak privacy
    epsilon = ∞    → no privacy (= baseline)

We also vary max_grad_norm (clipping threshold C) to show its effect.

Usage:
    python experiments/run_dp_sweep.py
    python experiments/run_dp_sweep.py --epochs 15 --quick
"""

import argparse
import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

from src.model import get_model
from src.utils import get_data_loaders, get_device, set_seed, logger, evaluate, MetricTracker
from src.privacy_engine import DPConfig, make_private


def parse_args():
    p = argparse.ArgumentParser(description="Sweep epsilon values for privacy-accuracy tradeoff")
    p.add_argument("--epochs",      type=int,   default=20,    help="Epochs per run")
    p.add_argument("--lr",          type=float, default=5e-4)
    p.add_argument("--batch-size",  type=int,   default=256)
    p.add_argument("--delta",       type=float, default=1e-5)
    p.add_argument("--data-dir",    type=str,   default="./data")
    p.add_argument("--results-dir", type=str,   default="./results")
    p.add_argument("--ckpt-dir",    type=str,   default="./checkpoints")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--quick",       action="store_true",
                   help="Use 20%% data subset and 5 epochs for fast testing")
    return p.parse_args()


def run_one_dp_experiment(
    target_epsilon: float,
    max_grad_norm: float,
    args,
    device: torch.device,
    train_loader,
    test_loader,
) -> dict:
    """Run one full DP training experiment and return summary dict."""
    set_seed(args.seed)

    model = get_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    config = DPConfig(
        target_epsilon=target_epsilon,
        target_delta=args.delta,
        max_grad_norm=max_grad_norm,
        num_epochs=args.epochs,
    )

    model, optimizer, train_loader_dp, privacy_engine = make_private(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        config=config,
    )

    sigma = optimizer.noise_multiplier
    logger.info(f"  σ={sigma:.4f} for ε={target_epsilon}")

    tracker = MetricTracker()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader_dp:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += images.size(0)

        tr_loss = total_loss / total
        tr_acc  = correct / total
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        eps_spent = privacy_engine.get_epsilon(args.delta)
        tracker.update(epoch, tr_loss, tr_acc, te_loss, te_acc, epsilon=eps_spent)

        if epoch % 5 == 0 or epoch == args.epochs:
            logger.info(
                f"    Epoch {epoch}/{args.epochs} | "
                f"test acc={te_acc:.3f} | ε spent={eps_spent:.3f}"
            )

    final_epsilon = privacy_engine.get_epsilon(args.delta)
    best_acc = tracker.best_test_acc()

    return {
        "target_epsilon":  target_epsilon,
        "final_epsilon":   round(final_epsilon, 4),
        "max_grad_norm":   max_grad_norm,
        "noise_multiplier": round(sigma, 4),
        "best_test_acc":   round(best_acc, 4),
        "epochs":          args.epochs,
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    if args.quick:
        args.epochs = 5
        subset = 0.2
        logger.info("Quick mode: 5 epochs, 20% data")
    else:
        subset = 1.0

    train_loader, test_loader = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        subset_fraction=subset,
    )

    # ── Experiment 1: Vary epsilon, fix C=1.0 ─────────────────────────────────
    epsilon_values = [0.5, 1.0, 2.0, 5.0, 10.0]
    clipping_norm  = 1.0

    logger.info("=" * 60)
    logger.info("  Experiment 1: Varying Epsilon (C=1.0 fixed)")
    logger.info("=" * 60)

    sweep_results = []
    for eps in epsilon_values:
        logger.info(f"\nRunning ε={eps}...")
        result = run_one_dp_experiment(eps, clipping_norm, args, device, train_loader, test_loader)
        sweep_results.append(result)
        logger.info(f"  ε={eps} → test acc={result['best_test_acc']:.4f}")

    # ── Experiment 2: Vary clipping norm, fix epsilon=2.0 ────────────────────
    clipping_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    fixed_epsilon   = 2.0

    logger.info("\n" + "=" * 60)
    logger.info("  Experiment 2: Varying Clipping Norm (ε=2.0 fixed)")
    logger.info("=" * 60)

    clipping_results = []
    for C in clipping_values:
        logger.info(f"\nRunning C={C}...")
        result = run_one_dp_experiment(fixed_epsilon, C, args, device, train_loader, test_loader)
        clipping_results.append(result)
        logger.info(f"  C={C} → test acc={result['best_test_acc']:.4f}")

    # ── Save results ──────────────────────────────────────────────────────────
    os.makedirs(args.results_dir, exist_ok=True)

    sweep_path = os.path.join(args.results_dir, "epsilon_sweep.json")
    with open(sweep_path, "w") as f:
        json.dump({"epsilon_sweep": sweep_results, "clipping_sweep": clipping_results}, f, indent=2)
    logger.info(f"\nResults saved → {sweep_path}")

    # Print summary table
    logger.info("\n  === Epsilon Sweep Summary ===")
    logger.info(f"  {'epsilon':>10} {'test_acc':>10} {'sigma':>10}")
    for r in sweep_results:
        logger.info(f"  {r['final_epsilon']:>10.3f} {r['best_test_acc']:>10.4f} {r['noise_multiplier']:>10.4f}")


if __name__ == "__main__":
    main()
