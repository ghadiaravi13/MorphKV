import json
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Load data ---
script_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(script_dir, "mkv_past_results", "mkv_default.json")) as f:
    mkv_default = json.load(f)
with open(os.path.join(script_dir, "mkv_fuse_unimp_ft_1.5_ib_0.5", "llama3.1-8b-instruct", "result.json")) as f:
    mkv_plus = json.load(f)

# --- Extract dataset short names and pair scores ---
# MorphKV default key pattern:  {dataset}_ws32_mc2000.0_morphkv_True_type_sum_fused_lenNone
# MorphKV_plus key pattern:     {dataset}_ws32_mc2000.0_ft2.0_morphkv_True_type_sum_fused_lenNone

DEFAULT_SUFFIX = "_ws32_mc2000.0_morphkv_True_type_sum_fused_lenNone"
PLUS_SUFFIX = "_ws32_mc2000.0_ft1.5_ao_False_morphkv_True_type_sum_fused_lenNone"

def extract_name(key, suffix):
    """Strip the known suffix to get the short dataset name."""
    if key.endswith(suffix):
        return key[: -len(suffix)]
    return key

# Build {short_name: score} dicts
default_scores = {extract_name(k, DEFAULT_SUFFIX): v for k, v in mkv_default.items()}
plus_scores = {extract_name(k, PLUS_SUFFIX): v for k, v in mkv_plus.items()}

# Only keep datasets present in both, and skip where default score is 0 (cannot normalize)
datasets = sorted(
    k for k in default_scores if k in plus_scores and default_scores[k] != 0.0
)

skipped = [k for k in default_scores if k in plus_scores and default_scores[k] == 0.0]
if skipped:
    print(f"Skipped (default score = 0, cannot normalize): {skipped}")

# Normalized scores: MorphKV default = 1.0, MorphKV_plus = ratio
norm_default = [1.0] * len(datasets)
norm_plus = [plus_scores[d] / default_scores[d] for d in datasets]

# --- Plot ---
fig, ax = plt.subplots(figsize=(16, 6))

x = np.arange(len(datasets))
bar_width = 0.35

bars_default = ax.bar(
    x - bar_width / 2, norm_default, bar_width,
    label="MorphKV (baseline)", color="#4C72B0", edgecolor="white", linewidth=0.6
)
bars_plus = ax.bar(
    x + bar_width / 2, norm_plus, bar_width,
    label="MorphKV+ (ft 1.5, ib 0.5)", color="#DD8452", edgecolor="white", linewidth=0.6
)

# Reference line at y = 1.0
ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

# Annotate ratio values on top of MorphKV+ bars
for i, (bar, val) in enumerate(zip(bars_plus, norm_plus)):
    ax.text(
        bar.get_x() + bar.get_width() / 2 + 0.01, bar.get_height() + 0.02,
        f"{val:.2f}", ha="center", va="bottom", fontsize=13, rotation=45
    )

ax.set_xlabel("Dataset", fontsize=12)
ax.set_ylabel("Normalized Score (MorphKV = 1.0)", fontsize=13)
ax.set_title("MorphKV vs MorphKV+ — Normalized LongBench Scores (Llama 3.1-8B-Instruct)", fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(datasets, rotation=45, ha="right", fontsize=13)
ax.legend(fontsize=10)
ax.set_ylim(0, max(norm_plus) + 0.3)

plt.tight_layout()

out_path = os.path.join(script_dir, "mkv_fuse_unimp_ft_1.5_ib_0.5", "normalized_scores.png")
plt.savefig(out_path, dpi=200)
print(f"Plot saved to {out_path}")
plt.show()
