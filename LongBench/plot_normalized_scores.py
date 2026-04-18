import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Dataset-specific keys begin with the dataset name and then append run metadata
# such as window size, ft, ao, pruning mode, and prerope flags.
KEY_SPLIT_TOKEN = "_ws32_mc2048.0"
BASELINE_DIR = "mkv_results_default"
COMPARISON_DIR = "hier_attn_cache"
OUTPUT_NAME = "normalized_scores.png"
SKIP_DATASETS = {} #{"dureader"}

# Raw scores stored as (full_attention, morphkv) tuples per dataset.
raw_scores_llama = {
    "2wikimqa":             (16.5, 14.9),
    "dureader":             (30.0, 22.5),
    "hotpotqa":             (16.7, 15.9),
    "multi_news":           (26.8, 26.6),
    "multifieldqa_en":      (27.4, 25.7),
    "multifieldqa_zh":      (20.1, 19.9),
    "musique":              (11.4, 10.7),
    "narrativeqa":          (32.0, 31.9),
    "passage_count":        ( 6.9,  7.5),
    "passage_retrieval_en": (97.7, 97.8),
    "qasper":               (13.2, 11.9),
    "qmsum":                (23.6, 23.6),
    "samsum":               (43.7, 42.9),
    "triviaqa":             (91.6, 91.5),
    "vcsum":                (16.1, 15.2),
}

raw_scores_mistral = {
    "2wikimqa":             (27.1, 26.7),
    "dureader":             (30.4, 23.9),
    "hotpotqa":             (43.0, 40.8),
    "multi_news":           (27.1, 26.6),
    "multifieldqa_en":      (49.2, 48.4),
    "multifieldqa_zh":      (48.3, 43.0),
    "musique":              (18.8, 16.7),
    "narrativeqa":          (26.7, 26.7),
    "passage_count":        ( 2.8,  3.0),
    "passage_retrieval_en": (87.0, 85.9),
    "qasper":               (33.0, 30.9),
    "qmsum":                (24.2, 23.6),
    "samsum":               (42.8, 42.3),
    "triviaqa":             (86.2, 86.3),
    "vcsum":                (15.2, 13.7),
}

raw_scores_phi4 = {
    "2wikimqa":             (22.2, 22.6),
    "dureader":             (29.0, 24.1),
    "hotpotqa":             (19.6, 19.3),
    "multi_news":           (25.9, 25.5),
    "multifieldqa_en":      (38.2, 38.2),
    "multifieldqa_zh":      (48.9, 46.4),
    "musique":              ( 6.0,  6.2),
    "narrativeqa":          (20.7, 21.0),
    "passage_count":        (11.6, 12.6),
    "passage_retrieval_en": (63.3, 64.3),
    "qasper":               (33.3, 31.2),
    "qmsum":                (22.9, 22.4),
    "samsum":               (48.2, 47.6),
    "triviaqa":             (90.4, 90.6),
    "vcsum":                (13.4, 12.3),
}

RAW_SCORES_BY_MODEL = {
    "llama": raw_scores_llama,
    "mistral": raw_scores_mistral,
    "phi4": raw_scores_phi4,
}


def get_raw_scores(model_name):
    """Return the raw (fullattn, morphkv) tuple dict for *model_name*, or None."""
    for key, scores in RAW_SCORES_BY_MODEL.items():
        if key in model_name.lower():
            return scores
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot normalized LongBench scores for MorphKV vs MorphKV++."
    )
    parser.add_argument(
        "model_name",
        help="Model folder name under mkv_results/ and hier_cache_ft2.0_ib0.8/.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Use hardcoded raw scores for MorphKV baseline and full attention "
        "instead of loading MorphKV baseline from BASELINE_DIR.",
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
    comparison_result_path = comparison_dir / "result.json"

    raw_score_tuples = get_raw_scores(args.model_name)

    if args.raw:
        if raw_score_tuples is None:
            raise ValueError(
                f"--raw requested but no hardcoded raw scores found for model '{args.model_name}'."
            )
        default_scores = {k: mkv for k, (_fa, mkv) in raw_score_tuples.items()}
        fullattn_raw = {k: fa for k, (fa, _mkv) in raw_score_tuples.items()}

        if not comparison_result_path.exists():
            raise FileNotFoundError(
                f"Missing required result.json: {comparison_result_path}"
            )
        mkv_plus = load_result(comparison_result_path)
        plus_scores = build_scores(mkv_plus)
    else:
        baseline_result_path = baseline_dir / "result.json"
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
        fullattn_raw = (
            {k: fa for k, (fa, _mkv) in raw_score_tuples.items()}
            if raw_score_tuples is not None
            else None
        )

    has_fullattn = fullattn_raw is not None
    if not has_fullattn:
        print(
            f"WARNING: No full-attention data found for model '{args.model_name}'. "
            "Plotting without full-attention bars."
        )

    datasets = sorted(
        key
        for key in default_scores
        if key in plus_scores
        and default_scores[key] != 0.0
        and key not in SKIP_DATASETS
        and (not has_fullattn or key in fullattn_raw)
    )
    if not datasets:
        raise ValueError(
            "No overlapping datasets found with non-zero baseline scores between the result files."
        )

    skipped = [
        key
        for key in default_scores
        if key in plus_scores and (default_scores[key] == 0.0 or key in SKIP_DATASETS)
    ]
    if has_fullattn:
        skipped += [
            key
            for key in default_scores
            if key in plus_scores and key not in fullattn_raw and key not in skipped
        ]
    if skipped:
        print(f"Skipped: {skipped}")

    norm_default = [1.0] * len(datasets)
    norm_plus = [plus_scores[dataset] / default_scores[dataset] for dataset in datasets]
    norm_fullattn = (
        [raw_score_tuples[dataset][0] / default_scores[dataset] for dataset in datasets] # raw_score_tuples[dataset][1]
        if has_fullattn
        else None
    )

    geomean = lambda vals: float(np.exp(np.mean(np.log(vals))))
    datasets.append("GeoMean")
    norm_default.append(1.0)
    norm_plus.append(geomean(norm_plus))
    if norm_fullattn is not None:
        norm_fullattn.append(geomean(norm_fullattn))

    fig, ax = plt.subplots(figsize=(20, 8))

    x = np.arange(len(datasets))
    bar_width = 0.25 if has_fullattn else 0.35

    if has_fullattn:
        offset_default = -bar_width
        offset_plus = 0
        offset_fullattn = bar_width
    else:
        offset_default = -bar_width / 2
        offset_plus = bar_width / 2

    ax.bar(
        x + offset_default,
        norm_default,
        bar_width,
        label="MorphKV (baseline)",
        color="#4C72B0",
        edgecolor="white",
        linewidth=0.6,
    )
    bars_plus = ax.bar(
        x + offset_plus,
        norm_plus,
        bar_width,
        label="MorphKV++ (ft 2.0, ib 0.8)",
        color="#DD8452",
        edgecolor="white",
        linewidth=0.6,
    )

    if has_fullattn:
        bars_fullattn = ax.bar(
            x + offset_fullattn,
            norm_fullattn,
            bar_width,
            label="Full Attention",
            color="#55A868",
            edgecolor="white",
            linewidth=0.6,
        )

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axvline(
        x=len(datasets) - 1.5, color="gray", linestyle=":", linewidth=1.0, alpha=0.5
    )

    for bar, value in zip(bars_plus, norm_plus):
        ax.text(
            bar.get_x() + bar.get_width() / 2 + 0.01,
            bar.get_height() + 0.02,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
            rotation=90,
        )

    if has_fullattn:
        for bar, value in zip(bars_fullattn, norm_fullattn):
            ax.text(
                bar.get_x() + bar.get_width() / 2 + 0.01,
                bar.get_height() + 0.02,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=11,
                rotation=90,
            )

    all_values = norm_plus + (norm_fullattn if has_fullattn else [])

    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel("Normalized Score (MorphKV = 1.0)", fontsize=13)
    ax.set_title(
        f"MorphKV vs MorphKV++ vs Full Attention — Normalized LongBench Scores ({args.model_name})",
        fontsize=13,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha="right", fontsize=13)
    ax.legend(fontsize=11, loc="upper right")
    ax.set_ylim(0, max(all_values) + 0.3)

    plt.tight_layout()

    output_paths = [baseline_dir / OUTPUT_NAME, comparison_dir / OUTPUT_NAME]
    for output_path in output_paths:
        fig.savefig(output_path, dpi=200)
        print(f"Plot saved to {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
