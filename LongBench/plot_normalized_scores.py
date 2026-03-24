import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Dataset-specific keys begin with the dataset name and then append run metadata
# such as window size, ft, ao, pruning mode, and prerope flags.
KEY_SPLIT_TOKEN = "_ws32_mc2000.0"
BASELINE_DIR = "mkv_results"
COMPARISON_DIR = "hier_cache_ft2.0_ib0.8"
OUTPUT_NAME = "normalized_scores.png"
SKIP_DATASETS = {"dureader"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot normalized LongBench scores for MorphKV vs MorphKV++."
    )
    parser.add_argument(
        "model_name",
        help="Model folder name under mkv_results/ and hier_cache_ft2.0_ib0.8/.",
    )
    return parser.parse_args()


def extract_name(key):
    """Extract the dataset short name from a result key."""
    if KEY_SPLIT_TOKEN in key:
        return key.split(KEY_SPLIT_TOKEN, 1)[0]
    return key


def load_result(result_path):
    with result_path.open() as f:
        return json.load(f)


def build_scores(results):
    return {extract_name(key): value for key, value in results.items()}


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    baseline_dir = script_dir / BASELINE_DIR / args.model_name
    comparison_dir = script_dir / COMPARISON_DIR / args.model_name
    baseline_result_path = baseline_dir / "result.json"
    comparison_result_path = comparison_dir / "result.json"

    missing_paths = [
        str(path)
        for path in (baseline_result_path, comparison_result_path)
        if not path.exists()
    ]
    if missing_paths:
        raise FileNotFoundError(
            "Missing required result.json file(s):\n" + "\n".join(missing_paths)
        )

    mkv_default = load_result(baseline_result_path)
    mkv_plus = load_result(comparison_result_path)

    default_scores = build_scores(mkv_default)
    plus_scores = build_scores(mkv_plus)

    datasets = sorted(
        key
        for key in default_scores
        if key in plus_scores
        and default_scores[key] != 0.0
        and key not in SKIP_DATASETS
    )
    if not datasets:
        raise ValueError(
            "No overlapping datasets found with non-zero baseline scores between the two result files."
        )

    skipped = [
        key
        for key in default_scores
        if key in plus_scores and (default_scores[key] == 0.0 or key in SKIP_DATASETS)
    ]
    if skipped:
        print(f"Skipped (default score = 0, cannot normalize): {skipped}")

    norm_default = [1.0] * len(datasets)
    norm_plus = [plus_scores[dataset] / default_scores[dataset] for dataset in datasets]

    fig, ax = plt.subplots(figsize=(16, 6))

    x = np.arange(len(datasets))
    bar_width = 0.35

    ax.bar(
        x - bar_width / 2,
        norm_default,
        bar_width,
        label="MorphKV (baseline)",
        color="#4C72B0",
        edgecolor="white",
        linewidth=0.6,
    )
    bars_plus = ax.bar(
        x + bar_width / 2,
        norm_plus,
        bar_width,
        label="MorphKV++ (ft 2.0, ib 0.8)",
        color="#DD8452",
        edgecolor="white",
        linewidth=0.6,
    )

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    for bar, value in zip(bars_plus, norm_plus):
        ax.text(
            bar.get_x() + bar.get_width() / 2 + 0.01,
            bar.get_height() + 0.02,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=13,
            rotation=45,
        )

    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel("Normalized Score (MorphKV = 1.0)", fontsize=13)
    ax.set_title(
        f"MorphKV vs MorphKV++ — Normalized LongBench Scores ({args.model_name})",
        fontsize=13,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha="right", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(norm_plus) + 0.3)

    plt.tight_layout()

    output_paths = [baseline_dir / OUTPUT_NAME, comparison_dir / OUTPUT_NAME]
    for output_path in output_paths:
        fig.savefig(output_path, dpi=200)
        print(f"Plot saved to {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
