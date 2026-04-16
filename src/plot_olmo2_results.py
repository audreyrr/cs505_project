"""
Visualize OLMo evaluation results.

Usage:

 python src/plot_olmo_results.py

 python src/plot_olmo_results.py --results \
   results/results_olmo2_1b_vllm.json \
   results/results_olmo2_7b_vllm.json \
   results/results_olmo2_13b_vllm.json

 python src/plot_olmo_results.py --output figures/olmo_accuracy.png
"""

import json
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent.parent


def extract_step(revision_key: str) -> int:
    """
    Extract training step from an OLMo revision key.

    Example:
      allenai/OLMo-2-0425-1B::stage1-step2000-tokens5B -> 2000
    """
    match = re.search(r"step(\d+)", revision_key)
    if not match:
        raise ValueError(f"Could not extract step from revision key: {revision_key}")
    return int(match.group(1))


def load_avg_accuracy(path: Path):
    with open(path) as f:
        data = json.load(f)

    results = data["results"]
    steps, accs = [], []

    for revision_key, task_accs in results.items():
        step = extract_step(revision_key)
        avg = np.mean(list(task_accs.values()))
        steps.append(step)
        accs.append(avg)

    # Sort by step
    pairs = sorted(zip(steps, accs))
    steps = [p[0] for p in pairs]
    accs = [p[1] for p in pairs]

    return steps, accs


def infer_label(path: Path) -> str:
    """
    Infer model size label from filename.
    Example:
      results_olmo2_1b_vllm.json -> 1b
    """
    stem = path.stem
    stem = stem.replace("results_olmo2_", "")
    stem = stem.replace("_vllm", "")
    return stem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        nargs="+",
        default=None,
        help="Result JSON files to plot. Defaults to all OLMo results/*.json.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="figures/olmo2_accuracy.png",
        help="Output path for the figure.",
    )
    args = parser.parse_args()

    result_paths = (
        [Path(p) for p in args.results]
        if args.results
        else sorted((ROOT / "results").glob("results_olmo2_*_vllm.json"))
    )

    if not result_paths:
        print("No OLMo result files found.")
        return

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    for path in result_paths:
        label = infer_label(path)
        steps, accs = load_avg_accuracy(path)
        ax.plot(steps, accs, label=label, linewidth=1.5)

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Average Accuracy (all tasks)", fontsize=12)
    ax.set_title("OLMo-2 — ICL Accuracy vs Training Step", fontsize=13)
    ax.legend(title="Model size", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()