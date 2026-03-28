"""
Feature engineering for TriML.

Adds rolling health/training features and computes:
  - ACWR  (acute:chronic workload ratio, 7d / 28d TSS)
  - HRV z-score  (deviation from personal 14-day baseline)
  - sleep_composite  (sleep_hours × sleep_quality, z-scored per athlete)
  - rhr_trend  (7-day linear slope of resting_hr — rising = overreach signal)
  - grit_score  (0–100 composite wellness metric)
  - load_class  (Overreaching / Balanced / Undertrained based on ACWR)

All rolling windows are computed *within each athlete* to prevent leakage.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Minimum history (days) before a row is usable for ML
MIN_HISTORY_DAYS = 28

# ACWR zone thresholds — from Gabbett (2016) BJSM literature
# Values above 1.3 indicate spike risk (Overreaching)
# Values below 0.8 indicate insufficient load (Undertrained)
ACWR_HIGH = 1.3   # above → Overreaching
ACWR_LOW  = 0.8   # below → Undertrained

# Load class labels and their integer codes
LOAD_CLASSES = ["Undertrained", "Balanced", "Overreaching"]  # 0, 1, 2

# Features used for model input
FEATURE_COLS = [
    # Rolling / computed
    "acwr",
    "hrv_zscore",
    "sleep_composite_z",
    "rhr_trend",
    # Daily wearable signals
    "body_battery_morning",
    "stress",
    "sleep_hours",
    "deep_sleep",
    "rem_sleep",
    "sleep_quality",
    "hrv",
    "resting_hr",
    # Training load
    "tss",
    "duration_minutes",
    "intensity_factor",
    "training_effect_aerobic",
    "training_effect_anaerobic",
    # Static athlete profile
    "age",
    "vo2max",
    "ftp",
    "training_experience",
    "weekly_training_hours",
    "gender_enc",
    "lifestyle_enc",
]


# ---------------------------------------------------------------------------
# Rolling helpers (applied per-athlete group)
# ---------------------------------------------------------------------------

def _rolling_slope(series: pd.Series, window: int = 7) -> pd.Series:
    """
    Per-element: fit OLS slope over the trailing `window` values.
    Returns NaN where fewer than 3 observations are available.
    """
    slopes = np.full(len(series), np.nan)
    vals = series.values
    for i in range(len(vals)):
        start = max(0, i - window + 1)
        chunk = vals[start : i + 1]
        chunk = chunk[~np.isnan(chunk)]
        if len(chunk) < 3:
            continue
        x = np.arange(len(chunk), dtype=float)
        # OLS slope via deviation from mean
        xm = x - x.mean()
        slopes[i] = np.dot(xm, chunk - chunk.mean()) / np.dot(xm, xm)
    return pd.Series(slopes, index=series.index)


def _engineer_athlete(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all rolling features for a single athlete's DataFrame.
    Expects df already sorted by date.
    """
    df = df.copy()

    # --- ACWR ---
    acute  = df["tss"].rolling(7,  min_periods=1).mean()
    chronic = df["tss"].rolling(28, min_periods=7).mean()
    df["acwr"] = (acute / chronic.replace(0, np.nan)).clip(0, 3)

    # --- HRV z-score (deviation from personal 14-day rolling baseline) ---
    hrv_14m = df["hrv"].rolling(14, min_periods=5).mean()
    hrv_14s = df["hrv"].rolling(14, min_periods=5).std().replace(0, np.nan)
    df["hrv_zscore"] = (df["hrv"] - hrv_14m) / hrv_14s

    # --- Sleep composite (hours × quality), z-scored per athlete ---
    raw_sleep = df["sleep_hours"] * df["sleep_quality"]
    mu = raw_sleep.mean()
    sd = raw_sleep.std()
    df["sleep_composite_z"] = (raw_sleep - mu) / (sd if sd > 0 else 1)

    # --- RHR 7-day trend (bpm/day; positive = rising = overreach signal) ---
    df["rhr_trend"] = _rolling_slope(df["resting_hr"], window=7)

    # --- Grit Score sub-scores (each 0–1, higher = MORE STRAIN / DANGER) ---
    # High Grit = you're grinding through fatigue. Low Grit = you're fresh.
    #
    # HRV dropping below baseline → HIGH grit (pushing through poor recovery)
    hrv_sub   = 1 / (1 + np.exp(df["hrv_zscore"]))   # flipped: negative z = high score

    # Poor sleep → HIGH grit (training despite bad recovery)
    sleep_sub = 1 / (1 + np.exp(df["sleep_composite_z"]))  # flipped: negative z = high score

    # Low battery → HIGH grit (body is depleted)
    bat_sub   = 1 - (df["body_battery_morning"] / 100).clip(0, 1)

    # High stress → HIGH grit
    stress_max = df["stress"].max() if df["stress"].max() > 0 else 1
    stress_sub = (df["stress"] / stress_max).clip(0, 1)

    # ACWR far from 1.0 (especially > 1.3) → HIGH grit (overreaching)
    acwr_sub  = (df["acwr"] - 1.0).abs().clip(0, 1)

    df["grit_score"] = 100 * (
        0.25 * hrv_sub
        + 0.25 * sleep_sub
        + 0.20 * bat_sub
        + 0.15 * stress_sub
        + 0.15 * acwr_sub
    )

    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def engineer_features(df_merged: pd.DataFrame) -> pd.DataFrame:
    """
    Add all engineered features to the merged DataFrame.

    Processes each athlete independently (no cross-athlete leakage).

    Args:
        df_merged: Output of build_merged(), 366,000 rows.

    Returns:
        Same DataFrame with additional feature columns + grit_score + load_class.
    """
    # Encode categoricals once (consistent codes across all athletes)
    gender_enc    = LabelEncoder().fit(df_merged["gender"])
    lifestyle_enc = LabelEncoder().fit(df_merged["lifestyle"])

    df_merged = df_merged.copy()
    df_merged["gender_enc"]    = gender_enc.transform(df_merged["gender"])
    df_merged["lifestyle_enc"] = lifestyle_enc.transform(df_merged["lifestyle"])

    # Apply rolling feature engineering per athlete
    parts = []
    for _, grp in df_merged.groupby("athlete_id", sort=False):
        grp_sorted = grp.sort_values("date")
        parts.append(_engineer_athlete(grp_sorted))

    out = pd.concat(parts).sort_values(["athlete_id", "date"]).reset_index(drop=True)

    # --- Load class from Grit Score distribution (proposal §3.1) ---
    # Thresholds derived from dataset percentiles so each class has
    # roughly equal representation; aligns with ACWR literature cutoffs.
    q_low  = out["grit_score"].quantile(0.25)
    q_high = out["grit_score"].quantile(0.75)
    # HIGH Grit Score → athlete is grinding / overreached    (class 2)
    # LOW Grit Score  → athlete is fresh / undertrained      (class 0)
    # Middle band     → optimal training balance             (class 1)
    def _grit_class(v):
        if pd.isna(v):
            return 1
        if v >= q_high:
            return 2  # Overreaching (high grit = high danger)
        if v <= q_low:
            return 0  # Undertrained (low grit = fresh, room to push)
        return 1      # Balanced

    out["load_class"] = out["grit_score"].apply(_grit_class)
    out["grit_q_low"]  = q_low   # store thresholds for reporting
    out["grit_q_high"] = q_high

    return out


def get_feature_matrix(df_feat: pd.DataFrame):
    """
    Extract model-ready numpy arrays from an engineered DataFrame.

    Drops the first MIN_HISTORY_DAYS rows per athlete (insufficient rolling history).
    Drops rows with any NaN in feature columns.

    Returns:
        X          : np.ndarray, shape (n_samples, n_features)
        y_class    : np.ndarray, shape (n_samples,)  — injury flag (0/1)
        y_grit     : np.ndarray, shape (n_samples,)  — grit_score (0–100)
        y_load     : np.ndarray, shape (n_samples,)  — load_class (0/1/2)
        groups     : np.ndarray, shape (n_samples,)  — athlete_id (for GroupKFold)
        feature_names : list[str]
    """
    # Drop first 28 days per athlete (not enough history for 28d chronic)
    def _drop_head(grp):
        return grp.iloc[MIN_HISTORY_DAYS:]

    df = df_feat.groupby("athlete_id", group_keys=False).apply(_drop_head)

    # Keep only rows with complete feature data
    needed = FEATURE_COLS + ["injury", "grit_score", "load_class", "athlete_id"]
    df = df[needed].dropna().reset_index(drop=True)

    X      = df[FEATURE_COLS].values.astype(np.float32)
    y_class = df["injury"].values.astype(np.int64)
    y_grit  = df["grit_score"].values.astype(np.float32)
    y_load  = df["load_class"].values.astype(np.int64)
    groups  = df["athlete_id"].values

    return X, y_class, y_grit, y_load, groups, FEATURE_COLS
