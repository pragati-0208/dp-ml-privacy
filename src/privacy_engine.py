"""
privacy_engine.py
-----------------
Clean wrapper around Opacus for managing DP training.

Why this wrapper?
- Centralises all Opacus-specific setup in one place
- Makes it easy to swap privacy accountants (RDP vs f-DP) later
- Provides utilities for computing sigma given (epsilon, delta, epochs, n)

Key concepts explained here so you can talk about them in interviews:

    Sigma (noise multiplier):
        The ratio of Gaussian noise std to the clipping threshold.
        sigma = noise_std / max_grad_norm
        Higher sigma = more noise = stronger privacy = lower utility.

    Privacy accountant (RDP accountant):
        Tracks accumulated privacy cost using Rényi Differential Privacy.
        Each gradient step spends some privacy budget.
        After T steps the accountant converts the total RDP cost to (epsilon, delta)-DP.

    Per-sample gradient clipping:
        Standard SGD clips the AVERAGE gradient after it's computed.
        DP-SGD clips each INDIVIDUAL sample's gradient before averaging.
        This bounds the sensitivity — the max influence any one sample can have.
        Opacus achieves this using functorch or hooks without materialising
        all per-sample gradients simultaneously (memory efficient).
"""

from dataclasses import dataclass
from typing import Tuple
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils import logger


@dataclass
class DPConfig:
    """
    Configuration for a DP training run.

    Attributes:
        target_epsilon: Privacy budget (ε). Lower is stronger privacy.
        target_delta:   Failure probability (δ). Typically 1/N where N = dataset size.
        max_grad_norm:  Per-sample gradient clipping threshold (C).
        num_epochs:     Number of training epochs (needed to compute sigma).

    The Gaussian mechanism guarantees:
        Pr[M(D) ∈ S] ≤ exp(ε) · Pr[M(D') ∈ S] + δ
    for any neighbouring datasets D, D' differing by one record.
    """
    target_epsilon: float = 10.0
    target_delta: float = 1e-5
    max_grad_norm: float = 1.0
    num_epochs: int = 20

    def __post_init__(self):
        if self.target_epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if not (0 < self.target_delta < 1):
            raise ValueError("delta must be in (0, 1)")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")

    def describe(self) -> str:
        return (
            f"DPConfig(ε={self.target_epsilon}, δ={self.target_delta:.1e}, "
            f"C={self.max_grad_norm}, epochs={self.num_epochs})"
        )


def compute_noise_multiplier(
    target_epsilon: float,
    target_delta: float,
    sample_rate: float,
    num_steps: int,
    accountant: str = "rdp",
) -> float:
    """
    Compute the noise multiplier σ needed to achieve (ε, δ)-DP.

    Uses Opacus's built-in accountant for the computation.
    This is a utility function for inspecting sigma without running training.

    Args:
        target_epsilon: Desired privacy budget.
        target_delta:   Desired delta.
        sample_rate:    Fraction of dataset per batch = batch_size / N.
        num_steps:      Total optimizer steps = num_epochs * (N / batch_size).
    """
    try:
        from opacus.accountants.utils import get_noise_multiplier
        sigma = get_noise_multiplier(
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            sample_rate=sample_rate,
            steps=num_steps,
            accountant=accountant,
        )
        return sigma
    except ImportError:
        logger.warning("Opacus not installed; cannot compute noise multiplier.")
        return float("nan")


def get_privacy_summary(
    config: DPConfig,
    dataset_size: int,
    batch_size: int,
) -> dict:
    """
    Returns a human-readable summary of the DP setup for a given config.
    Useful for logging and for the README.
    """
    num_steps = config.num_epochs * (dataset_size // batch_size)
    sample_rate = batch_size / dataset_size
    sigma = compute_noise_multiplier(
        config.target_epsilon,
        config.target_delta,
        sample_rate,
        num_steps,
    )
    return {
        "target_epsilon": config.target_epsilon,
        "target_delta": config.target_delta,
        "max_grad_norm": config.max_grad_norm,
        "num_steps": num_steps,
        "sample_rate": round(sample_rate, 5),
        "noise_multiplier_sigma": round(sigma, 4) if not math.isnan(sigma) else "N/A",
        "privacy_guarantee": (
            f"({config.target_epsilon:.1f}, {config.target_delta:.0e})-DP"
        ),
    }


def make_private(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    config: DPConfig,
):
    """
    Wraps model, optimizer, and loader with Opacus PrivacyEngine.

    Returns: (private_model, private_optimizer, private_loader, privacy_engine)
    """
    try:
        from opacus import PrivacyEngine
        from opacus.validators import ModuleValidator
    except ImportError as e:
        raise ImportError("Install opacus: pip install opacus") from e

    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        logger.warning(f"Auto-fixing model: {errors}")
        model = ModuleValidator.fix(model)

    privacy_engine = PrivacyEngine()

    private_model, private_optimizer, private_loader = (
        privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=config.num_epochs,
            target_epsilon=config.target_epsilon,
            target_delta=config.target_delta,
            max_grad_norm=config.max_grad_norm,
        )
    )

    sigma = private_optimizer.noise_multiplier
    logger.info(
        f"Privacy engine attached | σ={sigma:.4f} | {config.describe()}"
    )

    return private_model, private_optimizer, private_loader, privacy_engine
