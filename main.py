"""
main.py
-------
Single entry point for all experiments.

Usage:
    python main.py baseline              # Train non-DP model
    python main.py sweep                 # Run epsilon sweep
    python main.py attack                # Run membership inference attack
    python main.py all                   # Run everything in sequence
    python main.py all --quick           # Fast version (small data, few epochs)

After running, open notebooks/visualize_results.ipynb to generate all plots.
"""

import argparse
import subprocess
import sys
import os


EXPERIMENTS = {
    "baseline": "experiments/run_baseline.py",
    "sweep":    "experiments/run_dp_sweep.py",
    "attack":   "experiments/run_attack.py",
}


def run(script: str, extra_args: list = None):
    cmd = [sys.executable, script] + (extra_args or [])
    print(f"\n{'='*60}")
    print(f"  Running: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"ERROR: {script} failed with code {result.returncode}")
        sys.exit(result.returncode)


def main():
    p = argparse.ArgumentParser(description="DP-ML Privacy Experiments")
    p.add_argument("experiment", choices=["baseline", "sweep", "attack", "all"])
    p.add_argument("--quick",    action="store_true", help="Fast run for testing")
    p.add_argument("--epochs",   type=int, default=None)
    p.add_argument("--epsilon",  type=float, default=2.0, help="Epsilon for attack experiment")
    args = p.parse_args()

    extra = []
    if args.quick:   extra += ["--quick"]
    if args.epochs:  extra += ["--epochs", str(args.epochs)]

    os.makedirs("data",        exist_ok=True)
    os.makedirs("results",     exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    if args.experiment == "all":
        for name, script in EXPERIMENTS.items():
            e = extra.copy()
            if name == "attack":
                e += ["--epsilon", str(args.epsilon)]
            run(script, e)
        print("\n\nAll experiments complete!")
        print("Next step: jupyter notebook notebooks/visualize_results.ipynb")
    else:
        e = extra.copy()
        if args.experiment == "attack":
            e += ["--epsilon", str(args.epsilon)]
        run(EXPERIMENTS[args.experiment], e)


if __name__ == "__main__":
    main()
