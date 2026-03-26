"""
generate_plots.py
Generate publication-quality visualizations from TriML ML pipeline results.
All plots are saved as 300 DPI PNGs to results/plots/.
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
import seaborn as sns
from math import pi

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PKL = os.path.join(BASE_DIR, "results", "ml_results.pkl")
PLOTS_DIR = os.path.join(BASE_DIR, "results", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.style.use("dark_background")

DARK_BG   = "#0d1117"
PANEL_BG  = "#161b22"
GRID_COL  = "#30363d"
TEXT_COL  = "#e6edf3"
ACCENT1   = "#4A90D9"   # blue
ACCENT2   = "#D94040"   # red
ACCENT3   = "#F5C842"   # yellow
ACCENT4   = "#FFFFFF"   # white
ACCENT5   = "#4A90D9"   # blue (alias)

MODEL_COLORS = {
    "LR":    ACCENT1,    # blue
    "RF":    ACCENT2,    # red
    "DNN":   ACCENT3,    # yellow
    "Lasso": ACCENT4,    # white
}

RCPARAMS = {
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    PANEL_BG,
    "axes.edgecolor":    GRID_COL,
    "axes.labelcolor":   TEXT_COL,
    "axes.titlecolor":   TEXT_COL,
    "xtick.color":       TEXT_COL,
    "ytick.color":       TEXT_COL,
    "text.color":        TEXT_COL,
    "grid.color":        GRID_COL,
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "legend.facecolor":  PANEL_BG,
    "legend.edgecolor":  GRID_COL,
    "font.size":         11,
    "axes.titlesize":    16,
    "axes.labelsize":    12,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
}
plt.rcParams.update(RCPARAMS)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading results from", RESULTS_PKL)
with open(RESULTS_PKL, "rb") as f:
    payload = pickle.load(f)

raw          = payload["raw"]
tables       = payload["tables"]
feature_names = payload["feature_names"]
df_sample    = payload["df_feat_sample"].copy()

# ---------------------------------------------------------------------------
# Helper: save figure
# ---------------------------------------------------------------------------
def save(fig, fname):
    path = os.path.join(PLOTS_DIR, fname)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 01 — Injury classification grouped bar chart
# ---------------------------------------------------------------------------
def plot_01_injury_clf():
    metrics   = ["roc_auc", "f1_macro", "precision", "recall"]
    labels    = ["ROC-AUC", "F1-Macro", "Precision", "Recall"]
    models    = [("LR", raw["injury_clf"]["lr"]),
                 ("RF", raw["injury_clf"]["rf"]),
                 ("DNN", raw["injury_clf"]["mlp"])]

    n_metrics = len(metrics)
    n_models  = len(models)
    x         = np.arange(n_metrics)
    width     = 0.22
    offsets   = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=DARK_BG)
    ax.set_facecolor(PANEL_BG)

    for i, (name, data) in enumerate(models):
        means = [data["mean"][m] for m in metrics]
        stds  = [data["std"][m]  for m in metrics]
        bars  = ax.bar(x + offsets[i], means, width * 0.9,
                       label=name, color=MODEL_COLORS[name],
                       yerr=stds, capsize=4, error_kw={"ecolor": TEXT_COL, "alpha": 0.7},
                       alpha=0.88, zorder=3)
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.005,
                    f"{mean:.3f}", ha="center", va="bottom", fontsize=8,
                    color=TEXT_COL, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.6, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Injury Classification — Model Comparison", pad=14)
    ax.legend(loc="lower right")
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "01_injury_clf_comparison.png")


# ---------------------------------------------------------------------------
# Plot 02 — Load classification grouped bar chart
# ---------------------------------------------------------------------------
def plot_02_load_clf():
    metrics   = ["roc_auc", "f1_macro", "precision", "recall"]
    labels    = ["ROC-AUC", "F1-Macro", "Precision", "Recall"]
    models    = [("LR", raw["load_clf"]["lr"]),
                 ("RF", raw["load_clf"]["rf"]),
                 ("DNN", raw["load_clf"]["mlp"])]

    n_metrics = len(metrics)
    n_models  = len(models)
    x         = np.arange(n_metrics)
    width     = 0.22
    offsets   = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=DARK_BG)
    ax.set_facecolor(PANEL_BG)

    for i, (name, data) in enumerate(models):
        means = [data["mean"][m] for m in metrics]
        stds  = [data["std"][m]  for m in metrics]
        bars  = ax.bar(x + offsets[i], means, width * 0.9,
                       label=name, color=MODEL_COLORS[name],
                       yerr=stds, capsize=4, error_kw={"ecolor": TEXT_COL, "alpha": 0.7},
                       alpha=0.88, zorder=3)
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.003,
                    f"{mean:.3f}", ha="center", va="bottom", fontsize=8,
                    color=TEXT_COL, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.85, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Load Class Classification — Model Comparison", pad=14)
    ax.legend(loc="lower right")
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "02_load_clf_comparison.png")


# ---------------------------------------------------------------------------
# Plot 03 — Grit regression grouped bar chart (dual axis for RMSE/MAE vs R²)
# ---------------------------------------------------------------------------
def plot_03_grit_regression():
    models_info = [
        ("Lasso", raw["grit_reg"]["lasso"], ACCENT4),
        ("RF",    raw["grit_reg"]["rf"],    ACCENT5),
        ("DNN",   raw["grit_reg"]["mlp"],   ACCENT3),
    ]
    model_names = [m[0] for m in models_info]

    rmse_means = [m[1]["mean"]["rmse"] for m in models_info]
    rmse_stds  = [m[1]["std"]["rmse"]  for m in models_info]
    mae_means  = [m[1]["mean"]["mae"]  for m in models_info]
    mae_stds   = [m[1]["std"]["mae"]   for m in models_info]
    r2_means   = [m[1]["mean"]["r2"]   for m in models_info]
    r2_stds    = [m[1]["std"]["r2"]    for m in models_info]

    x = np.arange(len(model_names))
    width = 0.25

    fig, ax1 = plt.subplots(figsize=(10, 6), facecolor=DARK_BG)
    ax1.set_facecolor(PANEL_BG)
    ax2 = ax1.twinx()
    ax2.set_facecolor(PANEL_BG)

    b1 = ax1.bar(x - width, rmse_means, width * 0.9, label="RMSE",
                 color=ACCENT2, yerr=rmse_stds, capsize=4,
                 error_kw={"ecolor": TEXT_COL, "alpha": 0.7}, alpha=0.88, zorder=3)
    b2 = ax1.bar(x,          mae_means,  width * 0.9, label="MAE",
                 color=ACCENT1, yerr=mae_stds, capsize=4,
                 error_kw={"ecolor": TEXT_COL, "alpha": 0.7}, alpha=0.88, zorder=3)
    b3 = ax2.bar(x + width,  r2_means,   width * 0.9, label="R²",
                 color=ACCENT3, yerr=r2_stds, capsize=4,
                 error_kw={"ecolor": TEXT_COL, "alpha": 0.7}, alpha=0.88, zorder=3)

    for bars, vals in [(b1, rmse_means), (b2, mae_means)]:
        for bar, v in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=TEXT_COL)
    for bar, v in zip(b3, r2_means):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.0005,
                 f"{v:.4f}", ha="center", va="bottom", fontsize=8, color=TEXT_COL)

    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names)
    ax1.set_ylabel("RMSE / MAE (lower is better)", color=ACCENT2)
    ax2.set_ylabel("R² (higher is better)", color=ACCENT3)
    ax1.tick_params(axis="y", labelcolor=ACCENT2)
    ax2.tick_params(axis="y", labelcolor=ACCENT3)
    ax1.set_ylim(0, 2.4)
    ax2.set_ylim(0.96, 0.995)
    ax1.set_title("Grit Score Regression — Model Comparison", pad=14)

    handles = [b1, b2, b3]
    labels_leg = ["RMSE", "MAE", "R²"]
    ax1.legend(handles, labels_leg, loc="upper right")
    ax1.yaxis.grid(True, zorder=0)
    ax1.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "03_grit_regression_comparison.png")


# ---------------------------------------------------------------------------
# Plot 04 — Feature importance: Injury RF
# ---------------------------------------------------------------------------
FEATURE_CATEGORIES = {
    "deep_sleep":            ("sleep",    "#4A90D9"),   # blue
    "sleep_quality":         ("sleep",    "#4A90D9"),
    "sleep_hours":           ("sleep",    "#4A90D9"),
    "rem_sleep":             ("sleep",    "#4A90D9"),
    "light_sleep":           ("sleep",    "#4A90D9"),
    "sleep_composite_z":     ("sleep",    "#4A90D9"),
    "hrv_zscore":            ("hrv",      "#FFFFFF"),   # white
    "hrv":                   ("hrv",      "#FFFFFF"),
    "rhr_trend":             ("hrv",      "#FFFFFF"),
    "resting_hr":            ("hrv",      "#FFFFFF"),
    "acwr":                  ("training", "#F5C842"),   # yellow
    "tss":                   ("training", "#F5C842"),
    "duration_minutes":      ("training", "#F5C842"),
    "intensity_factor":      ("training", "#F5C842"),
    "training_effect_aerobic":   ("training", "#F5C842"),
    "training_effect_anaerobic": ("training", "#F5C842"),
    "weekly_training_hours": ("training", "#F5C842"),
    "stress":                ("recovery", "#D94040"),   # red
    "body_battery_morning":  ("recovery", "#D94040"),
    "age":                   ("static",   "#8b949e"),   # gray (neutral)
    "vo2max":                ("static",   "#8b949e"),
    "ftp":                   ("static",   "#8b949e"),
    "training_experience":   ("static",   "#8b949e"),
    "gender_enc":            ("static",   "#8b949e"),
    "lifestyle_enc":         ("static",   "#8b949e"),
}

CAT_LABELS = {
    "sleep":    "Sleep",
    "hrv":      "HRV / HR",
    "training": "Training Load",
    "recovery": "Stress & Recovery",
    "static":   "Static / Athlete",
}

def _feature_color(feat):
    if feat in FEATURE_CATEGORIES:
        return FEATURE_CATEGORIES[feat][1]
    return "#8b949e"

def _feature_cat(feat):
    if feat in FEATURE_CATEGORIES:
        return FEATURE_CATEGORIES[feat][0]
    return "static"


def plot_feature_importance(fi_df, title, fname):
    top10 = fi_df.head(10).copy()
    top10 = top10.sort_values("Importance", ascending=True)  # horizontal bar: bottom=lowest

    colors = [_feature_color(f) for f in top10["Feature"]]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=DARK_BG)
    ax.set_facecolor(PANEL_BG)

    bars = ax.barh(top10["Feature"], top10["Importance"],
                   color=colors, alpha=0.88, height=0.65, zorder=3)

    for bar, val in zip(bars, top10["Importance"]):
        ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=9, color=TEXT_COL)

    # Legend by category
    seen_cats = list(dict.fromkeys([_feature_cat(f) for f in top10["Feature"]]))
    cat_color_map = {
        "sleep":    "#4A90D9",   # blue
        "hrv":      "#FFFFFF",   # white
        "training": "#F5C842",   # yellow
        "recovery": "#D94040",   # red
        "static":   "#8b949e",   # gray
    }
    handles = [mpatches.Patch(color=cat_color_map[c], label=CAT_LABELS.get(c, c))
               for c in seen_cats if c in cat_color_map]
    ax.legend(handles=handles, loc="lower right", framealpha=0.6)

    ax.set_xlabel("Feature Importance")
    ax.set_title(title, pad=14)
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save(fig, fname)


def plot_04_fi_injury():
    plot_feature_importance(
        tables["fi_injury"],
        "Feature Importances — Injury Prediction (RF)",
        "04_feature_importance_injury.png"
    )


def plot_05_fi_grit():
    plot_feature_importance(
        tables["fi_grit"],
        "Feature Importances — Grit Score Regression (RF)",
        "05_feature_importance_grit.png"
    )


# ---------------------------------------------------------------------------
# Plot 06 — Radar chart: injury classification
# ---------------------------------------------------------------------------
def plot_06_radar():
    categories = ["AUC", "F1", "Precision", "Recall", "Accuracy"]
    n = len(categories)

    models_data = {
        "LR":  [raw["injury_clf"]["lr"]["mean"]["roc_auc"],
                raw["injury_clf"]["lr"]["mean"]["f1_macro"],
                raw["injury_clf"]["lr"]["mean"]["precision"],
                raw["injury_clf"]["lr"]["mean"]["recall"],
                raw["injury_clf"]["lr"]["mean"]["accuracy"]],
        "RF":  [raw["injury_clf"]["rf"]["mean"]["roc_auc"],
                raw["injury_clf"]["rf"]["mean"]["f1_macro"],
                raw["injury_clf"]["rf"]["mean"]["precision"],
                raw["injury_clf"]["rf"]["mean"]["recall"],
                raw["injury_clf"]["rf"]["mean"]["accuracy"]],
        "DNN": [raw["injury_clf"]["mlp"]["mean"]["roc_auc"],
                raw["injury_clf"]["mlp"]["mean"]["f1_macro"],
                raw["injury_clf"]["mlp"]["mean"]["precision"],
                raw["injury_clf"]["mlp"]["mean"]["recall"],
                raw["injury_clf"]["mlp"]["mean"]["accuracy"]],
    }

    angles = [i / float(n) * 2 * pi for i in range(n)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), facecolor=DARK_BG)
    ax.set_facecolor(PANEL_BG)

    for mname, vals in models_data.items():
        vals_plot = vals + vals[:1]
        color = MODEL_COLORS[mname]
        ax.plot(angles, vals_plot, "o-", linewidth=2, color=color, label=mname)
        ax.fill(angles, vals_plot, alpha=0.12, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12)
    ax.set_ylim(0.6, 1.0)
    ax.set_yticks([0.65, 0.75, 0.85, 0.95])
    ax.set_yticklabels(["0.65", "0.75", "0.85", "0.95"], size=9, color="#8b949e")
    ax.yaxis.grid(True, color=GRID_COL, linestyle="--", alpha=0.5)
    ax.xaxis.grid(True, color=GRID_COL, linestyle="--", alpha=0.5)
    ax.spines["polar"].set_color(GRID_COL)
    ax.set_title("Injury Classification — Radar Comparison", pad=22, size=16)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12))
    fig.tight_layout()
    save(fig, "06_model_summary_radar.png")


# ---------------------------------------------------------------------------
# Plot 07 — Grit score distribution
# ---------------------------------------------------------------------------
def plot_07_grit_dist():
    gs = df_sample["grit_score"].dropna()
    q25 = gs.quantile(0.25)
    q75 = gs.quantile(0.75)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=DARK_BG)
    ax.set_facecolor(PANEL_BG)

    # Shaded zones
    xmin, xmax = gs.min() - 2, gs.max() + 2
    ax.axvspan(xmin, q25,  alpha=0.12, color=ACCENT2,  label="Overreaching")
    ax.axvspan(q25,  q75,  alpha=0.12, color=ACCENT3,  label="Balanced")
    ax.axvspan(q75,  xmax, alpha=0.12, color=ACCENT1,  label="Undertrained")

    sns.histplot(gs, bins=50, kde=True, color=ACCENT4, alpha=0.65,
                 line_kws={"linewidth": 2}, ax=ax, zorder=3)

    ax.axvline(q25, color=ACCENT2, linewidth=2, linestyle="--",
               label=f"Q25 = {q25:.1f}", zorder=4)
    ax.axvline(q75, color=ACCENT1, linewidth=2, linestyle="--",
               label=f"Q75 = {q75:.1f}", zorder=4)

    ax.set_xlabel("Grit Score")
    ax.set_ylabel("Count")
    ax.set_title("Grit Score Distribution with Load Class Thresholds", pad=14)
    ax.legend(loc="upper right")
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "07_grit_score_distribution.png")


# ---------------------------------------------------------------------------
# Plot 08 — ACWR distribution
# ---------------------------------------------------------------------------
def plot_08_acwr_dist():
    acwr = df_sample["acwr"].dropna()

    LOW, HIGH = 0.8, 1.3

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=DARK_BG)
    ax.set_facecolor(PANEL_BG)

    xmin, xmax = acwr.min() - 0.05, acwr.max() + 0.05
    ax.axvspan(xmin, LOW,  alpha=0.14, color=ACCENT2, label="Under-training (<0.8)")
    ax.axvspan(LOW,  HIGH, alpha=0.10, color=ACCENT3, label="Optimal Zone (0.8–1.3)")
    ax.axvspan(HIGH, xmax, alpha=0.14, color=ACCENT1, label="Overreaching (>1.3)")

    sns.histplot(acwr, bins=50, kde=True, color=ACCENT5, alpha=0.70,
                 line_kws={"linewidth": 2}, ax=ax, zorder=3)

    ax.axvline(LOW,  color=ACCENT2, linewidth=2, linestyle="--",
               label=f"Threshold = {LOW}", zorder=4)
    ax.axvline(HIGH, color=ACCENT1, linewidth=2, linestyle="--",
               label=f"Threshold = {HIGH}", zorder=4)

    ax.set_xlabel("ACWR (Acute:Chronic Workload Ratio)")
    ax.set_ylabel("Count")
    ax.set_title("ACWR Distribution with Zone Thresholds", pad=14)
    ax.legend(loc="upper right")
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "08_acwr_distribution.png")


# ---------------------------------------------------------------------------
# Plot 09 — Violin plots: top 4 injury predictors split by injury
# ---------------------------------------------------------------------------
def plot_09_injury_violin():
    features = ["deep_sleep", "sleep_quality", "rhr_trend", "hrv_zscore"]
    feat_labels = ["Deep Sleep", "Sleep Quality", "RHR Trend", "HRV Z-score"]

    df_plot = df_sample[features + ["injury"]].dropna()
    df_plot["Injury"] = df_plot["injury"].map({0: "No Injury", 1: "Injury"})

    palette = {"No Injury": "#4A90D9", "Injury": "#D94040"}

    fig, axes = plt.subplots(1, 4, figsize=(16, 7), facecolor=DARK_BG)
    fig.subplots_adjust(wspace=0.35)

    for ax, feat, label in zip(axes, features, feat_labels):
        ax.set_facecolor(PANEL_BG)
        sns.violinplot(data=df_plot, x="Injury", y=feat, palette=palette,
                       inner="box", ax=ax, linewidth=1.2, cut=0)
        ax.set_title(label, fontsize=13)
        ax.set_xlabel("")
        ax.set_ylabel(label, fontsize=11)
        ax.yaxis.grid(True, zorder=0, alpha=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=10)

    fig.suptitle("Top Injury Predictors — Distribution by Injury Status",
                 fontsize=16, y=1.02)
    fig.tight_layout()
    save(fig, "09_injury_vs_features_violin.png")


# ---------------------------------------------------------------------------
# Plot 10 — Correlation heatmap
# ---------------------------------------------------------------------------
def plot_10_corr_heatmap():
    FEATURE_COLS = [
        "acwr", "hrv_zscore", "sleep_composite_z", "rhr_trend",
        "body_battery_morning", "stress", "sleep_hours", "deep_sleep",
        "rem_sleep", "sleep_quality", "hrv", "resting_hr",
        "tss", "duration_minutes", "intensity_factor",
        "training_effect_aerobic", "training_effect_anaerobic",
        "age", "vo2max", "ftp", "training_experience",
        "weekly_training_hours", "gender_enc", "lifestyle_enc",
    ]

    available = [c for c in FEATURE_COLS if c in df_sample.columns]
    corr = df_sample[available].corr()

    fig, ax = plt.subplots(figsize=(14, 12), facecolor=DARK_BG)
    ax.set_facecolor(PANEL_BG)

    cmap = sns.diverging_palette(220, 10, as_cmap=True)  # blue-to-red, colorblind safe
    mask = np.zeros_like(corr, dtype=bool)
    # No masking — show full matrix for publication

    sns.heatmap(
        corr, ax=ax, cmap=cmap, center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.3, linecolor=DARK_BG,
        annot=True, fmt=".2f", annot_kws={"size": 7},
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
    )

    ax.set_title("Feature Correlation Heatmap (24 Model Features)", pad=14)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0,  labelsize=9)
    fig.tight_layout()
    save(fig, "10_correlation_heatmap.png")


# ---------------------------------------------------------------------------
# Run all plots
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating plots...")
    plot_01_injury_clf()
    plot_02_load_clf()
    plot_03_grit_regression()
    plot_04_fi_injury()
    plot_05_fi_grit()
    plot_06_radar()
    plot_07_grit_dist()
    plot_08_acwr_dist()
    plot_09_injury_violin()
    plot_10_corr_heatmap()
    print("\nAll plots saved to:", PLOTS_DIR)

    # Verify
    expected = [
        "01_injury_clf_comparison.png",
        "02_load_clf_comparison.png",
        "03_grit_regression_comparison.png",
        "04_feature_importance_injury.png",
        "05_feature_importance_grit.png",
        "06_model_summary_radar.png",
        "07_grit_score_distribution.png",
        "08_acwr_distribution.png",
        "09_injury_vs_features_violin.png",
        "10_correlation_heatmap.png",
    ]
    missing = [f for f in expected if not os.path.exists(os.path.join(PLOTS_DIR, f))]
    if missing:
        print("\nWARNING: Missing files:", missing)
    else:
        print(f"\nAll {len(expected)} PNGs verified successfully.")
