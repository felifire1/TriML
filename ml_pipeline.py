#!/usr/bin/env python3
"""
ml_pipeline.py - Main ML pipeline for TriML
CS 6140 - Spring 2026

Runs the full pipeline:
  1. Load & merge the 3 CSVs
  2. Engineer features (ACWR, HRV z-score, Grit Score, etc.)
  3. Train 3 classifiers + 3 regressors with 5-fold GroupKFold CV
  4. Print results and save to pickle

Usage:
    python ml_pipeline.py                    # full run
    python ml_pipeline.py --sample 50        # quick test with 50 athletes
    python ml_pipeline.py --tune             # also run HP sweep
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.loader import (
    aggregate_activities, build_merged, ensure_data,
    load_activities, load_athletes, load_daily,
)
from src.features import engineer_features, get_feature_matrix, LOAD_CLASSES
from src.models import run_all_models, results_to_dataframes, hyperparameter_sweep


def _section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _print_clf_table(res, label):
    _section(f"Classification -- {label}")
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


def _print_reg_table(res, label):
    _section(f"Regression -- {label}")
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


def _print_feature_importances(fi, feature_names, label, top_n=10):
    _section(f"Top {top_n} Feature Importances (RF) -- {label}")
    ranked = sorted(zip(feature_names, fi), key=lambda x: x[1], reverse=True)
    for i, (fname, imp) in enumerate(ranked[:top_n], 1):
        bar = "#" * int(imp * 200)
        print(f"  {i:2}. {fname:<30} {imp:.4f}  {bar}")


def run(data_dir, results_dir, sample_n=None, tune=False):

    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. load data
    _section("Step 1/4 -- Loading data")
    t0 = time.time()

    data_dir = ensure_data(data_dir)
    print(f"  Data dir: {data_dir}")

    ath = load_athletes(data_dir / "athletes.csv")
    daily = load_daily(data_dir / "daily_data.csv")
    act = load_activities(data_dir / "activity_data.csv")

    if sample_n:
        sample_ids = ath["athlete_id"].sample(sample_n, random_state=42)
        ath = ath[ath["athlete_id"].isin(sample_ids)].copy()
        daily = daily[daily["athlete_id"].isin(sample_ids)].copy()
        act = act[act["athlete_id"].isin(sample_ids)].copy()
        print(f"  [sampling {sample_n} athletes for testing]")

    act_agg = aggregate_activities(act)
    merged = build_merged(daily, act_agg, ath)

    print(f"  Merged: {merged.shape}  ({time.time()-t0:.1f}s)")

    # 2. feature engineering
    _section("Step 2/4 -- Engineering features")
    t1 = time.time()

    df_feat = engineer_features(merged)

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
    print(f"\n  Injury prevalence: {injury_pct:.2f}%")
    print(f"  Done ({time.time()-t1:.1f}s)")

    # 3. build feature matrix
    _section("Step 3/4 -- Building feature matrix")
    X, y_injury, y_grit, y_load, groups, feat_names = get_feature_matrix(df_feat)

    print(f"  X: {X.shape}")
    print(f"  y_injury: {np.bincount(y_injury)}  (healthy / injured)")
    print(f"  y_load:   {np.bincount(y_load)}  (under / balanced / over)")
    print(f"  grit:     mean={y_grit.mean():.1f}  std={y_grit.std():.1f}")
    print(f"  Athletes: {len(np.unique(groups))}")

    # 4. train models
    _section("Step 4/4 -- Training (5-fold GroupKFold CV)")
    print("  This takes a while...\n")
    t2 = time.time()

    raw_results = run_all_models(X, y_injury, y_grit, y_load, groups, feat_names)

    print(f"\n  Done ({time.time()-t2:.1f}s)")

    # print results
    _print_clf_table(raw_results["injury_clf"], "Injury (binary)")
    _print_clf_table(raw_results["load_clf"], "Load Class (3-class)")
    _print_reg_table(raw_results["grit_reg"], "Grit Score (0-100)")

    _print_feature_importances(
        raw_results["injury_clf"]["feature_importance"], feat_names, "Injury"
    )
    _print_feature_importances(
        raw_results["grit_reg"]["feature_importance"], feat_names, "Grit Score"
    )

    # save CV results
    out_path = results_dir / "ml_results.pkl"
    payload = {
        "raw": raw_results,
        "tables": results_to_dataframes(raw_results),
        "feature_names": feat_names,
        "df_feat_sample": df_feat.sample(min(5000, len(df_feat)), random_state=42),
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    # train final models on all data for personal inference
    print("\n  Training final models on full dataset for inference...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    injury_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    injury_model.fit(X_scaled, y_injury)

    grit_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    grit_model.fit(X_scaled, y_grit)

    load_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    load_model.fit(X_scaled, y_load)

    models_path = results_dir / "trained_models.pkl"
    with open(models_path, "wb") as f:
        pickle.dump({
            "injury_clf": injury_model,
            "grit_reg": grit_model,
            "load_clf": load_model,
            "scaler": scaler,
            "feature_names": feat_names,
            "load_classes": LOAD_CLASSES,
        }, f)
    print(f"  Final models saved to {models_path}")

    # optional HP sweep
    if tune:
        run_hp_sweep(X, y_load, y_grit, groups, results_dir, n_classes=3)

    print(f"\n{'='*60}")
    print(f"  Saved to {out_path}")
    print(f"  Total: {time.time()-t0:.1f}s")
    print(f"{'='*60}")

    return payload


def run_hp_sweep(X, y_load, y_grit, groups, results_dir, n_classes=3):
    """Run HP sweep for all 6 models and save results."""
    _section("Hyperparameter Sweep (3-fold GroupKFold)")
    print("  Sweeping LR C, RF depth, DNN hidden size")
    print("  for both classification and regression\n")

    t0 = time.time()
    sweep = hyperparameter_sweep(X, y_load, y_grit, groups, n_classes=n_classes)

    # print tables
    sweep_info = [
        ("lr_C",           "LR -- C",              "C",           "mean_auc", "ROC-AUC"),
        ("rf_clf_depth",   "RF Clf -- max_depth",   "max_depth",   "mean_auc", "ROC-AUC"),
        ("dnn_clf_hidden", "DNN Clf -- hidden",     "hidden_size", "mean_auc", "ROC-AUC"),
        ("lasso_alpha",    "Lasso -- alpha",         "alpha",       "mean_r2",  "R²"),
        ("rf_reg_depth",   "RF Reg -- max_depth",   "max_depth",   "mean_r2",  "R²"),
        ("dnn_reg_hidden", "DNN Reg -- hidden",     "hidden_size", "mean_r2",  "R²"),
    ]

    for key, title, hp_col, metric_col, metric_name in sweep_info:
        df = sweep[key]
        std_col = metric_col.replace("mean", "std")
        _section(f"HP: {title}")
        print(f"  {hp_col:<15} {metric_name:>8}   std")
        print(f"  {'-'*35}")
        for _, row in df.iterrows():
            best = row[metric_col] == df[metric_col].max()
            flag = " <-- best" if best else ""
            print(f"  {str(row[hp_col]):<15} {row[metric_col]:>8.4f}  ±{row[std_col]:.4f}{flag}")

    sweep_path = results_dir / "hp_sweep.pkl"
    with open(sweep_path, "wb") as f:
        pickle.dump(sweep, f)

    print(f"\n  HP sweep done ({time.time()-t0:.1f}s) -- saved to {sweep_path}")
    return sweep


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TriML ML pipeline")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Dir with the 3 CSVs (default: auto-download from Zenodo)")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results",
                        help="Where to save outputs (default: results/)")
    parser.add_argument("--sample", type=int, default=None, metavar="N",
                        help="Only use N athletes (for quick testing)")
    parser.add_argument("--tune", action="store_true",
                        help="Also run hyperparameter sweep")
    args = parser.parse_args()

    run(data_dir=args.data_dir, results_dir=args.results_dir,
        sample_n=args.sample, tune=args.tune)
