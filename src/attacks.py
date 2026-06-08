"""
attacks.py
----------
Membership Inference Attack (MIA) implementation.

This is the crucial "so what?" of the entire project.
We don't just say "DP protects privacy" — we SHOW it by running an
actual attack and measuring how well it works with and without DP.

What is a Membership Inference Attack?
    Given a trained model and a data sample x, an attacker tries to determine:
    "Was x in the training set?"

    If the attacker can tell with high accuracy, the model is leaking
    information about its training data — a serious privacy violation
    (e.g., medical data, financial records).

How the attack works (Shokri et al., 2017):
    Intuition: models are typically MORE confident on training samples
    than on samples they haven't seen. So prediction confidence is a
    signal for membership.

    Simple threshold attack:
        1. Get model's predicted probability for the correct class.
        2. If confidence > threshold → predict "member"
        3. Otherwise → predict "non-member"

    Shadow model attack (implemented here):
        1. Train a "shadow model" on a different split of the data.
        2. For each sample, extract a feature vector (loss or softmax vector).
        3. Label training-set samples as "member", test-set samples as "non-member".
        4. Train a simple binary classifier (attack model) on these features.
        5. Apply the attack model to the TARGET model's predictions.

Metrics we report:
    - Attack accuracy: fraction of correct member/non-member predictions
    - TPR @ low FPR: true positive rate when false positive rate is ≤ 5%
      (industry standard — we care about low FPR regime)
    - AUC: area under the ROC curve
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from typing import Tuple, Dict
from tqdm import tqdm

from src.utils import logger


# ── Feature extraction ────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    feature_type: str = "loss",
) -> np.ndarray:
    """
    Extract features from model predictions for each sample.

    Args:
        feature_type: "loss"    → scalar cross-entropy loss per sample
                      "softmax" → full softmax probability vector (10-dim for CIFAR)
                      "conf"    → scalar max confidence (= max softmax prob)
    """
    model.eval()
    features = []
    criterion = nn.CrossEntropyLoss(reduction="none")

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        probs = F.softmax(logits, dim=1)

        if feature_type == "loss":
            feats = criterion(logits, labels).cpu().numpy()[:, None]  # (B, 1)
        elif feature_type == "softmax":
            feats = probs.cpu().numpy()                                # (B, 10)
        elif feature_type == "conf":
            feats = probs.max(dim=1).values.cpu().numpy()[:, None]    # (B, 1)
        else:
            raise ValueError(f"Unknown feature_type: {feature_type}")

        features.append(feats)

    return np.vstack(features)


# ── Threshold attack (simple baseline) ───────────────────────────────────────

def threshold_attack(
    model: nn.Module,
    member_loader: DataLoader,
    nonmember_loader: DataLoader,
    device: torch.device,
) -> Dict:
    """
    Simple threshold attack based on loss.

    Lower loss → model is more confident → more likely to be a member.
    Threshold is set to the loss value that maximises attack accuracy.
    """
    member_feats     = extract_features(model, member_loader, device, "loss")
    nonmember_feats  = extract_features(model, nonmember_loader, device, "loss")

    member_scores    = -member_feats.squeeze()     # negate: higher score = more likely member
    nonmember_scores = -nonmember_feats.squeeze()

    scores = np.concatenate([member_scores, nonmember_scores])
    labels = np.concatenate([
        np.ones(len(member_scores)),
        np.zeros(len(nonmember_scores))
    ])

    auc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)

    # TPR @ FPR ≤ 0.05
    tpr_at_low_fpr = tpr[fpr <= 0.05][-1] if any(fpr <= 0.05) else 0.0

    # Best accuracy over all thresholds
    best_acc = max(
        accuracy_score(labels, scores > t)
        for t in np.percentile(scores, np.linspace(0, 100, 100))
    )

    return {
        "attack_type": "threshold",
        "auc": float(auc),
        "tpr_at_5pct_fpr": float(tpr_at_low_fpr),
        "best_accuracy": float(best_acc),
        "member_mean_loss": float(-member_scores.mean()),
        "nonmember_mean_loss": float(-nonmember_scores.mean()),
    }


# ── Shadow model attack ────────────────────────────────────────────────────────

def shadow_model_attack(
    target_model: nn.Module,
    member_loader: DataLoader,
    nonmember_loader: DataLoader,
    shadow_member_loader: DataLoader,
    shadow_nonmember_loader: DataLoader,
    shadow_model: nn.Module,
    device: torch.device,
    feature_type: str = "softmax",
) -> Dict:
    """
    Full shadow model attack.

    Steps:
        1. Extract softmax features from shadow model (member + non-member splits)
        2. Train attack classifier on shadow model outputs
        3. Extract softmax features from target model
        4. Apply attack classifier → predict membership for target samples
    """
    logger.info("Extracting shadow model features...")
    shadow_member_feats    = extract_features(shadow_model, shadow_member_loader, device, feature_type)
    shadow_nonmember_feats = extract_features(shadow_model, shadow_nonmember_loader, device, feature_type)

    X_train = np.vstack([shadow_member_feats, shadow_nonmember_feats])
    y_train = np.concatenate([
        np.ones(len(shadow_member_feats)),
        np.zeros(len(shadow_nonmember_feats))
    ])

    # Simple logistic regression as attack model
    attack_clf = LogisticRegression(max_iter=1000, C=1.0)
    attack_clf.fit(X_train, y_train)
    logger.info("Attack classifier trained on shadow model outputs.")

    # Apply to target model
    logger.info("Extracting target model features...")
    target_member_feats    = extract_features(target_model, member_loader, device, feature_type)
    target_nonmember_feats = extract_features(target_model, nonmember_loader, device, feature_type)

    X_test = np.vstack([target_member_feats, target_nonmember_feats])
    y_test = np.concatenate([
        np.ones(len(target_member_feats)),
        np.zeros(len(target_nonmember_feats))
    ])

    scores = attack_clf.predict_proba(X_test)[:, 1]
    preds  = attack_clf.predict(X_test)

    auc = roc_auc_score(y_test, scores)
    fpr, tpr, _ = roc_curve(y_test, scores)
    tpr_at_low_fpr = tpr[fpr <= 0.05][-1] if any(fpr <= 0.05) else 0.0
    acc = accuracy_score(y_test, preds)

    return {
        "attack_type": "shadow_model",
        "auc": float(auc),
        "tpr_at_5pct_fpr": float(tpr_at_low_fpr),
        "best_accuracy": float(acc),
    }


# ── Evaluation helper ─────────────────────────────────────────────────────────

def evaluate_attack_vs_dp(
    results_no_dp: Dict,
    results_dp: Dict,
) -> str:
    """
    Returns a formatted string comparing attack results with and without DP.
    Useful for logging and the README.
    """
    lines = [
        "=" * 55,
        "  Membership Inference Attack Results",
        "=" * 55,
        f"{'Metric':<30} {'No DP':>10} {'With DP':>10}",
        "-" * 55,
    ]
    for key in ["auc", "tpr_at_5pct_fpr", "best_accuracy"]:
        label = {"auc": "AUC", "tpr_at_5pct_fpr": "TPR@5%FPR", "best_accuracy": "Accuracy"}[key]
        no_dp_val = results_no_dp.get(key, float("nan"))
        dp_val    = results_dp.get(key, float("nan"))
        lines.append(f"  {label:<28} {no_dp_val:>10.4f} {dp_val:>10.4f}")
    lines.append("=" * 55)
    lines.append("  Note: Values closer to 0.5 = better privacy")
    return "\n".join(lines)
