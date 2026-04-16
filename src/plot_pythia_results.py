"""
Visualize Pythia evaluation results.
Usage:
    python src/plot_results.py
    python src/plot_results.py --results results/results_70m.json results/results_160m.json
    python src/plot_results.py --output figures/accuracy.png
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent.parent


def load_avg_accuracy(path: Path):
    with open(path) as f:
        data = json.load(f)
    results = data["results"]

    steps, accs = [], []
    for step_str, task_accs in results.items():
        step = int(step_str)
        avg = np.mean(list(task_accs.values()))
        steps.append(step)
        accs.append(avg)

    # Sort by step
    pairs = sorted(zip(steps, accs))
    steps = [p[0] for p in pairs]
    accs  = [p[1] for p in pairs]
    return steps, accs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", default=None,
                        help="Result JSON files to plot. Defaults to all results/*.json.")
    parser.add_argument("--output", type=str, default="figures/accuracy.png",
                        help="Output path for the figure.")
    args = parser.parse_args()

    result_paths = [Path(p) for p in args.results] if args.results else sorted((ROOT / "results").glob("results_*.json"))
    if not result_paths:
        print("No result files found.")
        return

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    for path in result_paths:
        # Infer model size label from filename (e.g. results_70m.json -> 70m)
        label = path.stem.replace("results_", "")
        steps, accs = load_avg_accuracy(path)
        ax.plot(steps, accs, label=label, linewidth=1.5)

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Average Accuracy (all tasks)", fontsize=12)
    ax.set_title("Pythia Deduped — ICL Accuracy vs Training Step", fontsize=13)
    ax.legend(title="Model size", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
