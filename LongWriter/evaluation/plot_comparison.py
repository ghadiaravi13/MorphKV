#!/usr/bin/env python3
"""
Plot normalized bar charts comparing HierCache vs MorphKV (baseline)
for each model. All scores are normalized w.r.t. MorphKV results.
Outputs one PDF per model.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MKV_DIR = os.path.join(BASE_DIR, "mkv_results")
HIER_DIR = os.path.join(BASE_DIR, "hier_cache_ft2.0_ib0.8")
OUT_DIR = BASE_DIR

MODELS = ["phi4", "mistral", "llama3.1-8b-instruct", "qwen2.5"]

METRICS = [
    "Relevance",
    "Accuracy",
    "Coherence",
    "Clarity",
    "Breadth and Depth",
    "Reading Experience",
    "total",
]

DISPLAY_LABELS = [
    "Relevance",
    "Accuracy",
    "Coherence",
    "Clarity",
    "Breadth\n& Depth",
    "Reading\nExperience",
    "Total",
]

MODEL_DISPLAY = {
    "phi4": "Phi-4",
    "mistral": "Mistral",
    "llama3.1-8b-instruct": "Llama-3.1-8B-Instruct",
    "qwen2.5": "Qwen-2.5",
}


def load_scores(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    return next(iter(data.values()))


def main():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })

    for model in MODELS:
        mkv_path = os.path.join(MKV_DIR, model, "judge_result.json")
        hier_path = os.path.join(HIER_DIR, model, "judge_result.json")

        mkv_scores = load_scores(mkv_path)
        hier_scores = load_scores(hier_path)

        mkv_vals = np.array([mkv_scores[m] for m in METRICS])
        hier_vals = np.array([hier_scores[m] for m in METRICS])

        norm_hier = hier_vals / mkv_vals

        x = np.arange(len(METRICS))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 5))

        bars_mkv = ax.bar(
            x - width / 2, np.ones(len(METRICS)), width,
            label="MorphKV (baseline)", color="#5B8DB8", edgecolor="white", linewidth=0.8,
        )
        bars_hier = ax.bar(
            x + width / 2, norm_hier, width,
            label="HierCache", color="#E07B54", edgecolor="white", linewidth=0.8,
        )

        for bar, val in zip(bars_hier, norm_hier):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=8.5,
            )

        ax.axhline(y=1.0, color="grey", linewidth=0.6, linestyle="--")
        ax.set_ylabel("Normalized Score (MorphKV = 1.0)")
        ax.set_title(f"HierCache vs MorphKV — {MODEL_DISPLAY[model]}")
        ax.set_xticks(x)
        ax.set_xticklabels(DISPLAY_LABELS)
        ax.legend(loc="upper right", framealpha=0.9)
        ax.set_ylim(0, max(norm_hier.max(), 1.0) * 1.20)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        out_path = os.path.join(OUT_DIR, f"comparison_{model}.pdf")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
