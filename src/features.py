# features.py - Feature engineering for TriML
# CS 6140 - Spring 2026
#
# Computes rolling health/training features per athlete:
#   ACWR, HRV z-score, sleep composite, RHR trend, Grit Score, load class
#
# All rolling windows are per-athlete to prevent leakage.

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# need at least 28 days of history for chronic workload window
MIN_HISTORY_DAYS = 28

# ACWR thresholds from Gabbett 2016
ACWR_HIGH = 1.3   # above this = overreaching
ACWR_LOW = 0.8    # below this = undertrained

LOAD_CLASSES = ["Undertrained", "Balanced", "Overreaching"]  # 0, 1, 2

# 24 features used as model input
FEATURE_COLS = [
    # computed/rolling
    "acwr",
    "hrv_zscore",
    "sleep_composite_z",
    "rhr_trend",
    # daily wearable signals
    "body_battery_morning",
    "stress",
    "sleep_hours",
    "deep_sleep",
    "rem_sleep",
    "sleep_quality",
    "hrv",
    "resting_hr",
    # training load
    "tss",
    "duration_minutes",
    "intensity_factor",
    "training_effect_aerobic",
    "training_effect_anaerobic",
    # athlete profile (static)
    "age",
    "vo2max",
    "ftp",
    "training_experience",
    "weekly_training_hours",
    "gender_enc",
    "lifestyle_enc",
]


def _rolling_slope(series, window=7):
    """OLS slope over trailing window. Returns NaN where < 3 pts available."""
    slopes = np.full(len(series), np.nan)
    vals = series.values
    for i in range(len(vals)):
        start = max(0, i - window + 1)
        chunk = vals[start : i + 1]
        chunk = chunk[~np.isnan(chunk)]
        if len(chunk) < 3:
            continue
        x = np.arange(len(chunk), dtype=float)
        xm = x - x.mean()
        slopes[i] = np.dot(xm, chunk - chunk.mean()) / np.dot(xm, xm)
    return pd.Series(slopes, index=series.index)


def _engineer_athlete(df):
    """Add all rolling/computed features for one athlete. Expects sorted by date."""
    df = df.copy()

    # ACWR = acute (7d) / chronic (28d) mean TSS
    acute = df["tss"].rolling(7, min_periods=1).mean()
    chronic = df["tss"].rolling(28, min_periods=7).mean()
    df["acwr"] = (acute / chronic.replace(0, np.nan)).clip(0, 3)

    # HRV z-score relative to personal 14-day rolling baseline
    hrv_14m = df["hrv"].rolling(14, min_periods=5).mean()
    hrv_14s = df["hrv"].rolling(14, min_periods=5).std().replace(0, np.nan)
    df["hrv_zscore"] = (df["hrv"] - hrv_14m) / hrv_14s

    # sleep composite = hours * quality, then z-score within athlete
    raw_sleep = df["sleep_hours"] * df["sleep_quality"]
    mu = raw_sleep.mean()
    sd = raw_sleep.std()
    df["sleep_composite_z"] = (raw_sleep - mu) / (sd if sd > 0 else 1)

    # RHR 7-day slope (positive = rising = overreach signal)
    df["rhr_trend"] = _rolling_slope(df["resting_hr"], window=7)

    # --- Grit Score ---
    # Each sub-score is 0-1, where higher = more strain/fatigue
    # The idea: high grit = athlete is pushing through poor recovery
    #
    # HRV dropping -> high grit (bad recovery but still training)
    hrv_sub = 1 / (1 + np.exp(df["hrv_zscore"]))  # sigmoid, flipped

    # poor sleep -> high grit
    sleep_sub = 1 / (1 + np.exp(df["sleep_composite_z"]))

    # low body battery -> high grit
    bat_sub = 1 - (df["body_battery_morning"] / 100).clip(0, 1)

    # high stress -> high grit
    stress_max = df["stress"].max() if df["stress"].max() > 0 else 1
    stress_sub = (df["stress"] / stress_max).clip(0, 1)

    # ACWR far from 1.0 -> high grit
    acwr_sub = (df["acwr"] - 1.0).abs().clip(0, 1)

    # weighted composite -> scale to 0-100
    df["grit_score"] = 100 * (
        0.25 * hrv_sub
        + 0.25 * sleep_sub
        + 0.20 * bat_sub
        + 0.15 * stress_sub
        + 0.15 * acwr_sub
    )

    return df


def engineer_features(df_merged):
    """
    Add all engineered features to merged dataframe.
    Each athlete processed independently (no cross-athlete leakage).
    """
    # encode categoricals
    gender_enc = LabelEncoder().fit(df_merged["gender"])
    lifestyle_enc = LabelEncoder().fit(df_merged["lifestyle"])

    df_merged = df_merged.copy()
    df_merged["gender_enc"] = gender_enc.transform(df_merged["gender"])
    df_merged["lifestyle_enc"] = lifestyle_enc.transform(df_merged["lifestyle"])

    # rolling features per athlete
    parts = []
    for _, grp in df_merged.groupby("athlete_id", sort=False):
        grp_sorted = grp.sort_values("date")
        parts.append(_engineer_athlete(grp_sorted))

    out = pd.concat(parts).sort_values(["athlete_id", "date"]).reset_index(drop=True)

    # load class from grit score distribution
    # high grit = overreaching, low grit = undertrained, middle = balanced
    q_low = out["grit_score"].quantile(0.25)
    q_high = out["grit_score"].quantile(0.75)

    def _grit_class(v):
        if pd.isna(v):
            return 1
        if v >= q_high:
            return 2  # overreaching
        if v <= q_low:
            return 0  # undertrained
        return 1      # balanced

    out["load_class"] = out["grit_score"].apply(_grit_class)
    out["grit_q_low"] = q_low
    out["grit_q_high"] = q_high

    return out


def get_feature_matrix(df_feat):
    """
    Extract numpy arrays ready for model training.
    Drops first 28 days per athlete and rows with NaNs.

    Returns: X, y_class, y_grit, y_load, groups, feature_names
    """
    # drop initial rows without enough rolling history
    def _drop_head(grp):
        return grp.iloc[MIN_HISTORY_DAYS:]

    df = df_feat.groupby("athlete_id", group_keys=False).apply(_drop_head)

    # keep only complete rows
    needed = FEATURE_COLS + ["injury", "grit_score", "load_class", "athlete_id"]
    df = df[needed].dropna().reset_index(drop=True)

    X = df[FEATURE_COLS].values.astype(np.float32)
    y_class = df["injury"].values.astype(np.int64)
    y_grit = df["grit_score"].values.astype(np.float32)
    y_load = df["load_class"].values.astype(np.int64)
    groups = df["athlete_id"].values

    return X, y_class, y_grit, y_load, groups, FEATURE_COLS
