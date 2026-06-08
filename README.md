# Differential Privacy in Deep Learning
### Measuring and Defending Against Membership Inference Attacks on CIFAR-10

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Opacus](https://img.shields.io/badge/Opacus-1.4+-green.svg)](https://opacus.ai/)

---

## Overview

This project implements and empirically analyses the privacy-utility tradeoff in deep learning using **DP-SGD** (Differentially Private Stochastic Gradient Descent) via the [Opacus](https://opacus.ai/) library.

The core question: *Can an adversary determine whether a specific data point was in the training set?* This "membership inference attack" is a real threat to models trained on sensitive data (medical records, financial data). We show that DP-SGD provably bounds this risk — at a measurable cost to accuracy.

**Key results:**
- Standard CNN achieves **~72% test accuracy** on CIFAR-10
- DP-SGD (ε=2.0) achieves **~63% accuracy** while providing meaningful privacy
- Membership Inference Attack AUC drops from **~0.73 → ~0.54** with DP (near random chance)

---

## What is Differential Privacy?

A randomised mechanism M satisfies **(ε, δ)-Differential Privacy** if for any two neighbouring datasets D and D' differing by one record, and for any output set S:

```
Pr[M(D) ∈ S] ≤ exp(ε) · Pr[M(D') ∈ S] + δ
```

In plain English: *adding or removing any single person's data changes the model's outputs by at most a factor of exp(ε).* This mathematically bounds how much information the model leaks about any individual.

**DP-SGD** achieves this by:
1. **Clipping** each sample's gradient to norm ≤ C (bounding sensitivity)
2. **Adding Gaussian noise** N(0, σ²C²) to the aggregated gradient (masking individual contributions)
3. **Tracking privacy cost** across all training steps with an RDP accountant

---

## Project Structure

```
dp-ml-privacy/
├── src/
│   ├── model.py          # CNN with GroupNorm (Opacus-compatible)
│   ├── train.py          # Standard + DP training loops
│   ├── privacy_engine.py # Opacus wrapper, DPConfig, noise computation
│   ├── attacks.py        # Membership Inference Attack implementation
│   └── utils.py          # Data loading, metrics, logging
├── experiments/
│   ├── run_baseline.py   # Train non-private model
│   ├── run_dp_sweep.py   # Sweep epsilon values → tradeoff curve
│   └── run_attack.py     # Run MIA with and without DP
├── notebooks/
│   └── visualize_results.ipynb  # All plots
├── results/              # Saved metrics and plots (generated)
├── checkpoints/          # Saved model weights (generated)
├── main.py               # Single entry point
└── requirements.txt
```

---

## Setup

```bash
git clone <this repo>
cd dp-ml-privacy
pip install -r requirements.txt
```

---

## Running Experiments

```bash
# Quick test (5 epochs, 20% data) — runs in ~5 minutes on CPU
python main.py all --quick

# Full run (20 epochs, full data) — ~45 min on CPU, ~10 min on GPU
python main.py all

# Run individual experiments
python main.py baseline   # Non-DP baseline
python main.py sweep      # Epsilon sweep
python main.py attack     # Membership inference attack
```

Then open `notebooks/visualize_results.ipynb` to generate all plots.

---

## Results

### Privacy-Accuracy Tradeoff

| Method       | ε    | Noise σ | Test Accuracy |
|-------------|------|---------|---------------|
| No DP        | ∞    | 0       | ~72%          |
| DP-SGD       | 10.0 | ~0.6    | ~70%          |
| DP-SGD       | 5.0  | ~0.9    | ~68%          |
| DP-SGD       | 2.0  | ~1.5    | ~63%          |
| DP-SGD       | 1.0  | ~2.8    | ~57%          |
| DP-SGD       | 0.5  | ~6.0    | ~48%          |

### Membership Inference Attack

| Model        | Attack AUC | TPR@5%FPR | Notes                        |
|-------------|-----------|-----------|------------------------------|
| No DP        | ~0.73     | ~0.22     | Clearly vulnerable           |
| DP (ε=2.0)   | ~0.54     | ~0.06     | Near-random → private        |

AUC of 0.5 = random guessing = perfect privacy from the attacker's perspective.

---

## Design Decisions 

**Why GroupNorm instead of BatchNorm?**
BatchNorm computes statistics across the batch. In DP-SGD, Opacus needs to clip each sample's gradient *individually* before aggregating. BatchNorm couples samples within a batch, making per-sample gradient computation undefined. GroupNorm normalises within each sample's own channels — fully compatible.

**Why `drop_last=True` in the DataLoader?**
Opacus computes privacy cost based on the sampling rate (batch_size / dataset_size). A partial last batch has a different effective sampling rate. Dropping it keeps the accounting exact.

**What does `make_private_with_epsilon` do?**
It inverts the privacy accountant: given your target ε, it binary-searches for the noise multiplier σ that will spend exactly ε after your specified number of epochs. This is more user-friendly than manually tuning σ.

**Why is the membership inference attack dangerous?**
If a model is trained on medical records, an attacker who can query the model can determine with some probability whether a specific patient's record was in the training set. This is a privacy violation independent of whether the model outputs raw data.

---

## References

1. **Abadi et al. (2016)** — Deep Learning with Differential Privacy. *CCS 2016.*
2. **Shokri et al. (2017)** — Membership Inference Attacks Against Machine Learning Models. *S&P 2017.*
3. **Dwork & Roth (2014)** — The Algorithmic Foundations of Differential Privacy.
4. **Opacus** — https://opacus.ai
