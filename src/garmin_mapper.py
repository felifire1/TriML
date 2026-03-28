"""
Garmin data → TriML feature schema mapper.

Reads the 6 Garmin CSVs (daily_summary, sleep, hrv, activities_with_tss,
stress, body_battery) and produces a single DataFrame matching the
24-feature schema that the ML models expect.

Also computes the Grit Score (0–100) where:
    HIGH = dangerous (grinding through fatigue, overreaching)
    LOW  = safe (fresh, recovered, room to push)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GARMIN_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "garmin"


# ---------------------------------------------------------------------------
# Loaders — one per Garmin CSV
# ---------------------------------------------------------------------------

def _load_daily_summary(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "daily_summary.csv", parse_dates=["date"])
    return df[["date", "resting_hr", "avg_stress", "body_battery_high", "body_battery_low"]].rename(columns={
        "avg_stress": "stress",
        "body_battery_high": "body_battery_morning",  # best proxy: morning = highest
        "body_battery_low": "body_battery_evening",
    })


def _load_sleep(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "sleep.csv", parse_dates=["date"])
    # Convert seconds → hours
    for col in ["total_sleep_seconds", "deep_sleep_seconds", "light_sleep_seconds", "rem_sleep_seconds"]:
        df[col.replace("_seconds", "_hours")] = df[col] / 3600

    # Sleep score is a dict string like "{'value': 82, 'qualifierKey': 'GOOD'}"
    import ast
    def _parse_sleep_score(s):
        if pd.isna(s):
            return np.nan
        try:
            if isinstance(s, str) and s.startswith("{"):
                return float(ast.literal_eval(s)["value"])
            return float(s)
        except (ValueError, KeyError, SyntaxError):
            return np.nan

    df["sleep_quality"] = df["sleep_score"].apply(_parse_sleep_score) / 100

    return df[["date", "total_sleep_hours", "deep_sleep_hours",
               "light_sleep_hours", "rem_sleep_hours", "sleep_quality"]].rename(columns={
        "total_sleep_hours": "sleep_hours",
        "deep_sleep_hours": "deep_sleep",
        "light_sleep_hours": "light_sleep",
        "rem_sleep_hours": "rem_sleep",
    })


def _load_hrv(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "hrv.csv", parse_dates=["date"])
    # Use hrv_last_night if available, fall back to hrv_weekly_avg
    df["hrv"] = pd.to_numeric(df["hrv_last_night"], errors="coerce")
    if df["hrv"].isnull().all():
        df["hrv"] = pd.to_numeric(df["hrv_weekly_avg"], errors="coerce")
    return df[["date", "hrv"]]


def _load_activities(data_dir: Path) -> pd.DataFrame:
    """Load activities with TSS estimates, aggregate to daily."""
    path = data_dir / "activities_with_tss.csv"
    if not path.exists():
        path = data_dir / "activities.csv"
    df = pd.read_csv(path, parse_dates=["date"])

    # Map to canonical sports
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.tss_estimator import SPORT_MAP
    df["sport"] = df["activity_type"].map(SPORT_MAP).fillna("other")

    # Aggregate to daily
    daily_agg = df.groupby("date").agg(
        tss=("tss_estimated" if "tss_estimated" in df.columns else "tss", "sum"),
        duration_minutes=("duration_seconds", lambda x: x.sum() / 60),
        avg_hr_session=("avg_hr", "mean"),
        n_activities=("tss_estimated" if "tss_estimated" in df.columns else "tss", "count"),
        training_effect_aerobic=("training_effect_aerobic", "mean"),
        training_effect_anaerobic=("training_effect_anaerobic", "mean"),
    ).reset_index()

    # Fill NaN TSS with 0 (rest days will be handled in merge)
    daily_agg["tss"] = daily_agg["tss"].fillna(0)

    return daily_agg


def _load_stress(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "stress.csv", parse_dates=["date"])
    return df[["date", "avg_stress"]].rename(columns={"avg_stress": "stress_detail"})


def _load_body_battery(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "body_battery.csv", parse_dates=["date"])
    return df[["date", "morning_value", "evening_value"]].rename(columns={
        "morning_value": "bb_morning_detail",
        "evening_value": "bb_evening_detail",
    })


# ---------------------------------------------------------------------------
# Rolling feature computation (mirrors src/features.py logic)
# ---------------------------------------------------------------------------

def _rolling_slope(series: pd.Series, window: int = 7) -> pd.Series:
    """Linear slope over a rolling window (units per day)."""
    def _slope(arr):
        arr = arr.dropna()
        if len(arr) < 3:
            return np.nan
        x = np.arange(len(arr), dtype=float)
        return np.polyfit(x, arr.values, 1)[0]
    return series.rolling(window, min_periods=3).apply(_slope, raw=False)


def compute_grit_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all rolling features and Grit Score for a single athlete's daily data.

    Grit Score direction:
        HIGH (80–100) = Overreaching / dangerous
        MID  (40–79)  = Balanced
        LOW  (0–39)   = Undertrained / safe
    """
    df = df.sort_values("date").copy()

    # --- ACWR (Acute:Chronic Workload Ratio) ---
    acute  = df["tss"].rolling(7, min_periods=3).mean()
    chronic = df["tss"].rolling(28, min_periods=7).mean()
    df["acwr"] = (acute / chronic.replace(0, np.nan)).clip(0, 3)

    # --- HRV z-score (deviation from personal 14-day baseline) ---
    hrv_14m = df["hrv"].rolling(14, min_periods=5).mean()
    hrv_14s = df["hrv"].rolling(14, min_periods=5).std().replace(0, np.nan)
    df["hrv_zscore"] = (df["hrv"] - hrv_14m) / hrv_14s

    # --- Sleep composite (hours × quality), z-scored ---
    raw_sleep = df["sleep_hours"] * df["sleep_quality"]
    mu = raw_sleep.mean()
    sd = raw_sleep.std()
    df["sleep_composite_z"] = (raw_sleep - mu) / (sd if sd > 0 else 1)

    # --- RHR 7-day trend (bpm/day; positive = rising = overreach) ---
    df["rhr_trend"] = _rolling_slope(df["resting_hr"], window=7)

    # --- Grit Score sub-scores (each 0–1, higher = MORE STRAIN / DANGER) ---
    # HRV dropping below baseline → HIGH grit
    hrv_sub = 1 / (1 + np.exp(df["hrv_zscore"]))

    # Poor sleep → HIGH grit
    sleep_sub = 1 / (1 + np.exp(df["sleep_composite_z"]))

    # Low battery → HIGH grit
    bat_sub = 1 - (df["body_battery_morning"] / 100).clip(0, 1)

    # High stress → HIGH grit
    stress_max = df["stress"].max() if df["stress"].max() > 0 else 1
    stress_sub = (df["stress"] / stress_max).clip(0, 1)

    # ACWR far from 1.0 → HIGH grit
    acwr_sub = (df["acwr"] - 1.0).abs().clip(0, 1)

    df["grit_score"] = 100 * (
        0.25 * hrv_sub
        + 0.25 * sleep_sub
        + 0.20 * bat_sub
        + 0.15 * stress_sub
        + 0.15 * acwr_sub
    )

    # --- Load class ---
    def _grit_class(v):
        if pd.isna(v):
            return "Balanced"
        if v >= 70:
            return "Overreaching"
        if v <= 40:
            return "Undertrained"
        return "Balanced"

    df["load_class"] = df["grit_score"].apply(_grit_class)

    return df


# ---------------------------------------------------------------------------
# Main mapper
# ---------------------------------------------------------------------------

def build_personal_dataset(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Merge all Garmin CSVs into a single daily DataFrame with all features
    needed for ML inference + Grit Score.

    Returns:
        DataFrame with one row per day, including:
        - Raw signals: hrv, resting_hr, sleep_hours, deep_sleep, stress, etc.
        - Rolling features: acwr, hrv_zscore, sleep_composite_z, rhr_trend
        - Grit Score (0–100, high = dangerous)
        - Load class (Overreaching / Balanced / Undertrained)
    """
    data_dir = Path(data_dir) if data_dir else GARMIN_DATA_DIR

    print("Loading Garmin CSVs...")
    daily = _load_daily_summary(data_dir)
    sleep = _load_sleep(data_dir)
    hrv = _load_hrv(data_dir)
    activities = _load_activities(data_dir)
    stress = _load_stress(data_dir)
    battery = _load_body_battery(data_dir)

    # Merge all on date
    print("Merging into single daily table...")
    df = daily.merge(sleep, on="date", how="outer")
    df = df.merge(hrv, on="date", how="outer")
    df = df.merge(activities, on="date", how="left")
    df = df.merge(stress, on="date", how="left")
    df = df.merge(battery, on="date", how="left")

    # Use detailed stress/battery if available, fall back to daily_summary values
    if "stress_detail" in df.columns:
        df["stress"] = df["stress_detail"].fillna(df["stress"])
    if "bb_morning_detail" in df.columns:
        df["body_battery_morning"] = df["bb_morning_detail"].fillna(df["body_battery_morning"])
    if "bb_evening_detail" in df.columns:
        df["body_battery_evening"] = df["bb_evening_detail"].fillna(df["body_battery_evening"])

    # Rest days: TSS = 0
    df["tss"] = df["tss"].fillna(0)
    df["n_activities"] = df["n_activities"].fillna(0)

    # Forward-fill missing health metrics (watch not worn some days)
    health_cols = ["resting_hr", "hrv", "stress", "body_battery_morning",
                   "body_battery_evening", "sleep_hours", "deep_sleep",
                   "rem_sleep", "light_sleep", "sleep_quality"]
    df = df.sort_values("date")
    for col in health_cols:
        if col in df.columns:
            df[col] = df[col].ffill()

    print(f"  {len(df)} days, {df.date.min().date()} → {df.date.max().date()}")
    null_pct = df[health_cols].isnull().mean() * 100
    print(f"  Null % after forward-fill:")
    for col in health_cols:
        if col in df.columns:
            pct = df[col].isnull().mean() * 100
            if pct > 0:
                print(f"    {col}: {pct:.1f}%")

    # Drop rows with no health data at all (beginning of range)
    df = df.dropna(subset=["hrv", "resting_hr", "sleep_hours"]).reset_index(drop=True)

    # Compute rolling features + Grit Score
    print("Computing Grit Score and rolling features...")
    df = compute_grit_features(df)

    print(f"\n  Final dataset: {len(df)} days")
    gs = df["grit_score"].dropna()
    print(f"  Grit Score: mean={gs.mean():.1f}, std={gs.std():.1f}, "
          f"min={gs.min():.1f}, max={gs.max():.1f}")
    print(f"  Load class distribution:")
    for cls, count in df["load_class"].value_counts().items():
        pct = 100 * count / len(df)
        print(f"    {cls:<15} {count:>4} days ({pct:.1f}%)")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = build_personal_dataset()

    out_path = GARMIN_DATA_DIR / "personal_daily_features.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")

    # Show recent days
    print("\n=== Last 14 days ===")
    recent = df.tail(14)[["date", "hrv", "resting_hr", "sleep_hours",
                           "tss", "acwr", "hrv_zscore", "grit_score", "load_class"]]
    recent["date"] = recent["date"].dt.strftime("%m/%d")
    print(recent.to_string(index=False, float_format="%.1f"))
