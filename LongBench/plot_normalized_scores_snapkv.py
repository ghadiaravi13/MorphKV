import json
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Load data ---
script_dir = os.path.dirname(os.path.abspath(__file__))

with open("/home/rhg659/MorphKV/LongBench/mkv_results/mistral/result.json") as f:
    mkv_default = json.load(f)

pred_path = "val_mag_scoring_ft_2.0_ib_0.8/mistral"
with open(os.path.join(script_dir, pred_path, "result.json")) as f:
    mkv_plus = json.load(f)

# --- Extract dataset short names and pair scores ---
# MorphKV default key pattern:  {dataset}_ws32_mc2000.0_morphkv_True_type_sum_fused_lenNone
# MorphKV_plus key pattern:     {dataset}_ws32_mc2000.0_ft2.0_morphkv_True_type_sum_fused_lenNone

DEFAULT_SUFFIX = "_ws32_mc2000.0_morphkv_True_type_sum_fused_lenNone"
PLUS_SUFFIX = "_ws32_mc2000.0_ft2.0_ao_False_morphkv_True_type_unimp_sum_fused_lenNone"

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

"""["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
                    "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
                    "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]"""
snapkv_scores = {"narrativeqa": 0.958, "qasper": 1.0314, "multifieldqa_en": 1.008, "multifieldqa_zh": 0.9604, "2wikimqa": 0.9962, "dureader": 0.9916, "hotpotqa": 0.9926, "multi_news": 0.977, "musique": 1.0958, "passage_count": 0.985, "passage_retrieval_en": 0.833, "qmsum": 1.003, "samsum": 1.0085, "triviaqa": 0.9905, "vcsum": 1.0}
norm_snapkv = [snapkv_scores[d] if d in snapkv_scores else np.nan for d in datasets]

# --- Plot ---
fig, ax = plt.subplots(figsize=(16, 6))

x = np.arange(len(datasets)) * 1.5
bar_width = 0.35

bars_default = ax.bar(
    x - bar_width, norm_default, bar_width,
    label="MorphKV (baseline)", color="#4C72B0", edgecolor="white", linewidth=0.6
)
bars_plus = ax.bar(
    x, norm_plus, bar_width,
    label="MorphKV+ (ft 2.0, ib 0.8)", color="#DD8452", edgecolor="white", linewidth=0.6
)
bars_snapkv = ax.bar(
    x + bar_width, norm_snapkv, bar_width,
    label="SnapKV", color="#FFD700", edgecolor="white", linewidth=0.6
)
# Reference line at y = 1.0
ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

# Annotate ratio values on top of MorphKV+ bars
for i, (bar, val) in enumerate(zip(bars_plus, norm_plus)):
    ax.text(
        bar.get_x() + bar.get_width() / 2 + 0.01, bar.get_height() + 0.02,
        f"{val:.2f}", ha="center", va="bottom", fontsize=13, rotation=90
    )

for i, (bar, val) in enumerate(zip(bars_snapkv, norm_snapkv)):
    if not np.isnan(val):
        ax.text(
            bar.get_x() + bar.get_width() / 2 + 0.01, bar.get_height() + 0.02,
            f"{val:.2f}", ha="center", va="bottom", fontsize=13, rotation=90
        )

ax.set_xlabel("Dataset", fontsize=12)
ax.set_ylabel("Normalized Score (MorphKV = 1.0)", fontsize=13)
ax.set_title("MorphKV vs MorphKV+ — Normalized LongBench Scores (Mistral 7B)", fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(datasets, rotation=45, ha="right", fontsize=13)
ax.legend(fontsize=10)
ax.set_ylim(0, max(norm_plus) + 0.3)

plt.tight_layout()

out_path = os.path.join(script_dir, pred_path, "normalized_scores.png")
plt.savefig(out_path, dpi=200)
print(f"Plot saved to {out_path}")
plt.show()

# --- Interactive Plotly plot ---
import plotly.graph_objects as go

snapkv_clean = [v if not np.isnan(v) else None for v in norm_snapkv]
fmt = lambda vals: [f"{v:.2f}" if v is not None else "" for v in vals]

fig_plotly = go.Figure()

fig_plotly.add_trace(go.Bar(
    name="MorphKV (baseline)", x=datasets, y=norm_default,
    text=fmt(norm_default), textposition="outside", textangle=-45,
    marker_color="#4C72B0", marker_line=dict(color="white", width=0.6),
))
fig_plotly.add_trace(go.Bar(
    name="MorphKV+ (ft 2.0, ib 0.8)", x=datasets, y=norm_plus,
    text=fmt(norm_plus), textposition="outside", textangle=-45,
    marker_color="#DD8452", marker_line=dict(color="white", width=0.6),
))
fig_plotly.add_trace(go.Bar(
    name="SnapKV", x=datasets, y=snapkv_clean,
    text=fmt(snapkv_clean), textposition="outside", textangle=-45,
    marker_color="#FFD700", marker_line=dict(color="white", width=0.6),
))

fig_plotly.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.7)

fig_plotly.update_layout(
    barmode="group",
    title="MorphKV vs MorphKV+ — Normalized LongBench Scores (Mistral 7B)",
    xaxis_title="Dataset",
    yaxis_title="Normalized Score (MorphKV = 1.0)",
    yaxis_range=[0, max(norm_plus) + 0.3],
    xaxis_tickangle=-45,
    legend=dict(font=dict(size=12)),
    width=1200, height=600,
    template="plotly_white",
)

html_path = os.path.join(script_dir, pred_path, "normalized_scores.html")
fig_plotly.write_html(html_path)
print(f"Interactive plot saved to {html_path}")
