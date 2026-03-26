"""
TriML — Standalone ML Pipeline
================================
Runs the full machine learning pipeline end-to-end:

  1. Load & merge data (athletes + daily + activities)
  2. Engineer features:
       - ACWR (7d / 28d TSS)
       - HRV z-score (vs 14-day personal baseline)
       - Sleep composite (hours × quality, z-scored per athlete)
       - RHR 7-day trend (bpm/day linear slope)
       - Grit Score (0–100 composite wellness metric)
       - Load class (Undertrained / Balanced / Overreaching via ACWR)

  3. Train & evaluate with 5-fold GroupKFold CV (groups = athlete_id):
       Classification targets:
         - injury (0/1)          → LR, Random Forest, DNN
         - load_class (0/1/2)    → LR, Random Forest, DNN
       Regression targets:
         - grit_score (0–100)    → Lasso+Poly, Random Forest, DNN

  4. Print results tables + save to results/ml_results.pkl

Usage:
    python ml_pipeline.py [--data-dir PATH] [--results-dir PATH] [--sample N] [--tune]

    --data-dir    : directory containing the 3 CSVs (default: auto-detect / Zenodo)
    --results-dir : where to save outputs (default: results/)
    --sample N    : use only N athletes for a quick smoke-test (default: all 1000)
    --tune        : also run hyperparameter sweep (adds ~10–20 min)
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Make sure src/ is importable regardless of where script is called from
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.loader import (
    aggregate_activities,
    build_merged,
    ensure_data,
    load_activities,
    load_athletes,
    load_daily,
)
from src.features import engineer_features, get_feature_matrix, LOAD_CLASSES
from src.models import run_all_models, results_to_dataframes, hyperparameter_sweep


# ---------------------------------------------------------------------------
# Pretty printing helpers
# ---------------------------------------------------------------------------

SEP = "─" * 70

def _section(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def _print_clf_table(res: dict, label: str):
    _section(f"Classification — {label}")
    model_keys = [("lr", "Logistic Regression"), ("rf", "Random Forest"), ("mlp", "DNN (MLP)")]
    header = f"{'Model':<22} {'ROC-AUC':>12} {'F1-macro':>10} {'Precision':>11} {'Recall':>9} {'Accuracy':>10}"
    print(header)
    print("-" * len(header))
    for key, name in model_keys:
        m = res[key]["mean"]
        s = res[key]["std"]
        print(
            f"{name:<22} "
            f"{m['roc_auc']:>6.3f}±{s['roc_auc']:.3f}  "
            f"{m['f1_macro']:>5.3f}±{s['f1_macro']:.3f}  "
            f"{m['precision']:>5.3f}±{s['precision']:.3f}  "
            f"{m['recall']:>5.3f}±{s['recall']:.3f}  "
            f"{m['accuracy']:>5.3f}±{s['accuracy']:.3f}"
        )


def _print_reg_table(res: dict, label: str):
    _section(f"Regression — {label}")
    model_keys = [("lasso", "Lasso + Poly"), ("rf", "Random Forest"), ("mlp", "DNN (MLP)")]
    header = f"{'Model':<22} {'RMSE':>12} {'MAE':>12} {'R²':>12}"
    print(header)
    print("-" * len(header))
    for key, name in model_keys:
        m = res[key]["mean"]
        s = res[key]["std"]
        print(
            f"{name:<22} "
            f"{m['rmse']:>6.3f}±{s['rmse']:.3f}  "
            f"{m['mae']:>6.3f}±{s['mae']:.3f}  "
            f"{m['r2']:>6.3f}±{s['r2']:.3f}"
        )


def _print_feature_importances(fi: np.ndarray, feature_names: list, label: str, top_n: int = 10):
    _section(f"Top {top_n} Feature Importances (Random Forest) — {label}")
    ranked = sorted(zip(feature_names, fi), key=lambda x: x[1], reverse=True)
    for i, (fname, imp) in enumerate(ranked[:top_n], 1):
        bar = "█" * int(imp * 200)
        print(f"  {i:2}. {fname:<30} {imp:.4f}  {bar}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(data_dir: Path, results_dir: Path, sample_n=None, tune: bool = False):

    results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    _section("Step 1/4 — Loading data")
    t0 = time.time()

    data_dir = ensure_data(data_dir)
    print(f"  Data directory: {data_dir}")

    ath   = load_athletes(data_dir / "athletes.csv")
    daily = load_daily(data_dir / "daily_data.csv")
    act   = load_activities(data_dir / "activity_data.csv")

    if sample_n:
        sample_ids = ath["athlete_id"].sample(sample_n, random_state=42)
        ath   = ath[ath["athlete_id"].isin(sample_ids)].copy()
        daily = daily[daily["athlete_id"].isin(sample_ids)].copy()
        act   = act[act["athlete_id"].isin(sample_ids)].copy()
        print(f"  [--sample] Using {sample_n} athletes")

    act_agg = aggregate_activities(act)
    merged  = build_merged(daily, act_agg, ath)

    print(f"  Merged shape: {merged.shape}  ({time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------
    # 2. Feature engineering
    # ------------------------------------------------------------------
    _section("Step 2/4 — Engineering features")
    t1 = time.time()

    df_feat = engineer_features(merged)

    # Quick summary of engineered features
    for col in ("acwr", "hrv_zscore", "sleep_composite_z", "rhr_trend", "grit_score"):
        s = df_feat[col].dropna()
        print(f"  {col:<22} mean={s.mean():7.3f}  std={s.std():6.3f}  "
              f"min={s.min():7.3f}  max={s.max():7.3f}")

    load_dist = df_feat["load_class"].value_counts().sort_index()
    print(f"\n  Load class distribution:")
    for cls_id, count in load_dist.items():
        pct = 100 * count / len(df_feat)
        print(f"    {LOAD_CLASSES[cls_id]:<15} {count:>7,}  ({pct:.1f}%)")

    injury_pct = 100 * df_feat["injury"].mean()
    print(f"\n  Injury prevalence: {injury_pct:.2f}%  (class imbalance ratio)")
    print(f"  Feature engineering done  ({time.time()-t1:.1f}s)")

    # ------------------------------------------------------------------
    # 3. Build feature matrix
    # ------------------------------------------------------------------
    _section("Step 3/4 — Building feature matrix")
    X, y_injury, y_grit, y_load, groups, feat_names = get_feature_matrix(df_feat)

    print(f"  X shape      : {X.shape}")
    print(f"  y_injury     : {np.bincount(y_injury)}  (0=healthy, 1=injured)")
    print(f"  y_load       : {np.bincount(y_load)}  (0=Undertrained, 1=Balanced, 2=Overreaching)")
    print(f"  grit_score   : mean={y_grit.mean():.1f}  std={y_grit.std():.1f}")
    print(f"  Unique groups: {len(np.unique(groups))} athletes")

    # ------------------------------------------------------------------
    # 4. Train & evaluate
    # ------------------------------------------------------------------
    _section("Step 4/4 — Training models (5-fold GroupKFold CV)")
    print("  This may take 5–15 minutes depending on hardware...\n")
    t2 = time.time()

    raw_results = run_all_models(X, y_injury, y_grit, y_load, groups, feat_names)

    print(f"\n  All models done  ({time.time()-t2:.1f}s)")

    # ------------------------------------------------------------------
    # 5. Print results
    # ------------------------------------------------------------------
    _print_clf_table(raw_results["injury_clf"], "Injury (binary: 0=healthy, 1=injured)")
    _print_clf_table(raw_results["load_clf"],   "Load Class (0=Undertrained, 1=Balanced, 2=Overreaching)")
    _print_reg_table(raw_results["grit_reg"],   "Grit Score (0–100)")

    _print_feature_importances(
        raw_results["injury_clf"]["feature_importance"], feat_names, "Injury Classification"
    )
    _print_feature_importances(
        raw_results["grit_reg"]["feature_importance"], feat_names, "Grit Score Regression"
    )

    # ------------------------------------------------------------------
    # 6. Save results
    # ------------------------------------------------------------------
    out_path = results_dir / "ml_results.pkl"
    payload = {
        "raw": raw_results,
        "tables": results_to_dataframes(raw_results),
        "feature_names": feat_names,
        "df_feat_sample": df_feat.sample(min(5000, len(df_feat)), random_state=42),
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    # ------------------------------------------------------------------
    # 6 (optional). Hyperparameter sweep
    # ------------------------------------------------------------------
    if tune:
        run_hp_sweep(X, y_load, y_grit, groups, results_dir, n_classes=3)

    print(f"\n{SEP}")
    print(f"  Results saved → {out_path}")
    print(f"  Total runtime : {time.time()-t0:.1f}s")
    print(SEP)

    return payload


def run_hp_sweep(
    X: np.ndarray,
    y_load: np.ndarray,
    y_grit: np.ndarray,
    groups: np.ndarray,
    results_dir: Path,
    n_classes: int = 3,
):
    """
    Run hyperparameter sweep for all 6 models and save + print results.
    Investigates how each key HP affects CV performance (course requirement).
    """
    _section("Hyperparameter Sweep (3-fold GroupKFold)")
    print("  Sweeping: LR C · RF max_depth · DNN hidden_size")
    print("  for both classification (load_class) and regression (grit_score)\n")

    t0 = time.time()
    sweep = hyperparameter_sweep(X, y_load, y_grit, groups, n_classes=n_classes)

    # ---------- Print tables ----------
    sweep_info = [
        ("lr_C",          "LR — C (regularization)",      "C",           "mean_auc", "ROC-AUC"),
        ("rf_clf_depth",  "RF Classifier — max_depth",    "max_depth",   "mean_auc", "ROC-AUC"),
        ("dnn_clf_hidden","DNN Classifier — hidden size",  "hidden_size", "mean_auc", "ROC-AUC"),
        ("lasso_alpha",   "Lasso — alpha",                 "alpha",       "mean_r2",  "R²"),
        ("rf_reg_depth",  "RF Regressor — max_depth",     "max_depth",   "mean_r2",  "R²"),
        ("dnn_reg_hidden","DNN Regressor — hidden size",   "hidden_size", "mean_r2",  "R²"),
    ]

    for key, title, hp_col, metric_col, metric_name in sweep_info:
        df = sweep[key]
        std_col = metric_col.replace("mean", "std")
        _section(f"HP Sweep: {title}")
        print(f"  {hp_col:<15} {metric_name:>8}   std")
        print(f"  {'─'*35}")
        for _, row in df.iterrows():
            best = row[metric_col] == df[metric_col].max()
            flag = " ◄ best" if best else ""
            print(f"  {str(row[hp_col]):<15} {row[metric_col]:>8.4f}  ±{row[std_col]:.4f}{flag}")

    # Save
    sweep_path = results_dir / "hp_sweep.pkl"
    with open(sweep_path, "wb") as f:
        pickle.dump(sweep, f)

    print(f"\n  HP sweep done ({time.time()-t0:.1f}s) — saved → {sweep_path}")
    return sweep


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TriML ML pipeline")
    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help="Directory with athletes.csv / daily_data.csv / activity_data.csv "
             "(default: auto-detect project root, download from Zenodo if missing)",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=ROOT / "results",
        help="Where to save ml_results.pkl  (default: ./results/)",
    )
    parser.add_argument(
        "--sample", type=int, default=None, metavar="N",
        help="Use only N athletes (for quick smoke-tests; default: all 1000)",
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Also run hyperparameter sweep for all 6 models (adds ~10-20 min)",
    )
    args = parser.parse_args()

    run(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        sample_n=args.sample,
        tune=args.tune,
    )
