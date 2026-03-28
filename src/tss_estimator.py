"""
TSS (Training Stress Score) estimator for real Garmin data.

Estimates TSS for every workout using the best available data:
  1. Power-based TSS  (cycling/running with power meter)
  2. HR-based hrTSS   (any sport with heart rate)
  3. Duration-based    (fallback for yoga, strength, etc.)

References:
  - Coggan power-based TSS: (sec × NP × IF) / (FTP × 3600) × 100
  - hrTSS (Friel):  (sec × HRr × 0.64 × e^(1.92 × HRr)) / (male_factor × LTHR × 3600) × 100
    where HRr = avg_hr / LTHR
  - Duration-based:  sport-specific TSS/hour estimates
"""

import numpy as np
import pandas as pd
from typing import Optional


# ---------------------------------------------------------------------------
# Default athlete thresholds (can be overridden per-athlete)
# ---------------------------------------------------------------------------

DEFAULT_FTP = 275       # watts — Felipe's actual bike FTP
DEFAULT_LTHR = 176      # bpm — estimated 88% of 200 max HR
DEFAULT_MAX_HR = 200    # bpm
DEFAULT_RUN_THRESHOLD_PACE = 427.0  # sec/mile — 7:07/mile threshold pace
DEFAULT_SWIM_CSS = 1.064  # m/s — 1:34/100m CSS


# Sport-specific TSS/hour for duration-only fallback
DURATION_TSS_PER_HOUR = {
    "strength_training": 50,
    "yoga": 20,
    "hiking": 40,
    "other": 35,
    "rowing_v2": 55,
    "indoor_rowing": 55,
    "pickleball": 45,
    "hiit": 80,
}


# ---------------------------------------------------------------------------
# TSS calculation methods
# ---------------------------------------------------------------------------

def power_tss(duration_sec: float, avg_power: float, ftp: float) -> float:
    """
    Power-based TSS (Coggan formula).

    Assumes avg_power ≈ NP for simplicity. For more accuracy,
    normalized power from the activity file would be used.

    TSS = (duration × avg_power × IF) / (FTP × 3600) × 100
    where IF = avg_power / FTP
    """
    if ftp <= 0 or duration_sec <= 0 or avg_power <= 0:
        return 0.0
    intensity_factor = avg_power / ftp
    tss = (duration_sec * avg_power * intensity_factor) / (ftp * 3600) * 100
    return round(tss, 1)


def hr_tss(duration_sec: float, avg_hr: float, lthr: float) -> float:
    """
    Heart rate-based TSS (hrTSS) using the Friel/Coggan method.

    hrTSS = (duration × HRr × 0.64 × e^(1.92 × HRr)) / (k × LTHR × 3600) × 100
    where:
      HRr = avg_hr / LTHR  (heart rate ratio)
      k ≈ 1.0 (gender factor, simplification)
    """
    if lthr <= 0 or duration_sec <= 0 or avg_hr <= 0:
        return 0.0
    hr_ratio = avg_hr / lthr
    # Clamp to avoid extreme values from bad HR data
    hr_ratio = np.clip(hr_ratio, 0.5, 1.3)
    tss = (duration_sec * hr_ratio * 0.64 * np.exp(1.92 * hr_ratio)) / (3600) * 100
    # Scale relative to threshold effort (1 hour at LTHR ≈ 100 TSS)
    # Normalize so that HRr=1.0 for 1 hour = ~100 TSS
    one_hour_at_threshold = (3600 * 1.0 * 0.64 * np.exp(1.92 * 1.0)) / 3600 * 100
    tss = tss / one_hour_at_threshold * 100
    return round(tss, 1)


def duration_tss(duration_sec: float, sport: str) -> float:
    """
    Fallback: estimate TSS from duration using sport-specific TSS/hour rates.
    """
    if duration_sec <= 0:
        return 0.0
    hours = duration_sec / 3600
    rate = DURATION_TSS_PER_HOUR.get(sport, 40)  # default 40 TSS/hour
    return round(hours * rate, 1)


def run_tss(duration_sec: float, distance_m: float, avg_hr: float,
            lthr: float,
            threshold_pace_sec_mile: float = DEFAULT_RUN_THRESHOLD_PACE) -> float:
    """
    Running TSS using pace-based method (rTSS).

    rTSS = (duration/3600) × (threshold_pace / actual_pace)² × 100
    where pace is in sec/mile (lower = faster = higher intensity).

    Falls back to HR-based if no distance data.
    """
    if distance_m and distance_m > 0 and duration_sec > 0:
        # Convert to sec/mile
        miles = distance_m / 1609.34
        actual_pace_sec_mile = duration_sec / miles if miles > 0 else 9999
        # IF = threshold_pace / actual_pace (faster actual = higher IF)
        intensity = threshold_pace_sec_mile / actual_pace_sec_mile
        intensity = np.clip(intensity, 0.3, 1.5)
        # Standard rTSS: (duration/3600) × IF² × 100
        # The IF² mirrors Coggan's power TSS (NP/FTP)²
        hours = duration_sec / 3600
        return round(hours * (intensity ** 2) * 100, 1)

    # Fallback to HR
    if avg_hr and avg_hr > 0 and lthr > 0:
        return hr_tss(duration_sec, avg_hr, lthr)

    return duration_tss(duration_sec, "running")


def swim_tss(duration_sec: float, distance_m: float, avg_hr: float,
             lthr: float, css: float = DEFAULT_SWIM_CSS) -> float:
    """
    Swimming TSS: use HR-based if HR available, otherwise pace-based estimate.

    For pace-based: TSS ≈ (duration/3600) × (actual_speed / css)² × 100
    """
    if avg_hr and avg_hr > 0 and lthr > 0:
        return hr_tss(duration_sec, avg_hr, lthr)

    if distance_m and distance_m > 0 and duration_sec > 0:
        actual_speed = distance_m / duration_sec  # m/s
        intensity = actual_speed / css
        intensity = np.clip(intensity, 0.5, 1.3)
        hours = duration_sec / 3600
        return round(hours * (intensity ** 2) * 100, 1)

    return duration_tss(duration_sec, "lap_swimming")


# ---------------------------------------------------------------------------
# Sport classification (map Garmin activity types → canonical sports)
# ---------------------------------------------------------------------------

SPORT_MAP = {
    # Cycling
    "cycling": "bike",
    "virtual_ride": "bike",
    "road_biking": "bike",
    "indoor_cycling": "bike",
    "mountain_biking": "bike",
    "gravel_cycling": "bike",
    # Running
    "running": "run",
    "treadmill_running": "run",
    "trail_running": "run",
    "track_running": "run",
    # Swimming
    "lap_swimming": "swim",
    "open_water_swimming": "swim",
    # Strength
    "strength_training": "strength",
    "hiit": "strength",
    # Other (keep as-is)
    "yoga": "other",
    "hiking": "other",
    "rowing_v2": "other",
    "indoor_rowing": "other",
    "pickleball": "other",
    "other": "other",
}


# ---------------------------------------------------------------------------
# Main estimator
# ---------------------------------------------------------------------------

def estimate_tss(
    row: pd.Series,
    ftp: float = DEFAULT_FTP,
    lthr: float = DEFAULT_LTHR,
    css: float = DEFAULT_SWIM_CSS,
    run_threshold_pace: float = DEFAULT_RUN_THRESHOLD_PACE,
) -> float:
    """
    Estimate TSS for a single activity row using the best available method.

    Priority:
      1. If Garmin provided TSS → use it
      2. Cycling with power → power_tss()
      3. Running → pace-based run_tss() (7:07/mile threshold)
      4. Swimming → swim_tss() (HR or pace)
      5. Any sport with HR → hr_tss()
      6. Fallback → duration_tss()
    """
    sport = row.get("activity_type", "other")
    canonical = SPORT_MAP.get(sport, "other")
    dur = float(row.get("duration_seconds", 0) or 0)
    avg_power = float(row.get("avg_power", 0) or 0)
    avg_heart = float(row.get("avg_hr", 0) or 0)
    distance = float(row.get("distance_meters", 0) or 0)

    # 1. Garmin-provided TSS (rare but possible with some head units)
    garmin_tss = row.get("tss")
    if pd.notna(garmin_tss) and float(garmin_tss) > 0:
        return float(garmin_tss)

    # 2. Cycling with power
    if canonical == "bike" and avg_power > 0:
        return power_tss(dur, avg_power, ftp)

    # 3. Running → pace-based (more reliable than run power)
    if canonical == "run":
        return run_tss(dur, distance, avg_heart, lthr, run_threshold_pace)

    # 4. Swimming
    if canonical == "swim":
        return swim_tss(dur, distance, avg_heart, lthr, css)

    # 5. HR-based for any sport
    if avg_heart > 0:
        return hr_tss(dur, avg_heart, lthr)

    # 6. Duration fallback
    return duration_tss(dur, sport)


def estimate_all_tss(
    df_activities: pd.DataFrame,
    ftp: float = DEFAULT_FTP,
    lthr: float = DEFAULT_LTHR,
    css: float = DEFAULT_SWIM_CSS,
    run_threshold_pace: float = DEFAULT_RUN_THRESHOLD_PACE,
) -> pd.DataFrame:
    """
    Add estimated TSS to all activities. Returns a copy with new columns:
      - tss_estimated: the TSS value
      - tss_method: which method was used (power/hr/duration/swim/garmin)
      - sport_canonical: mapped to bike/run/swim/strength/other
    """
    df = df_activities.copy()

    tss_values = []
    methods = []

    for _, row in df.iterrows():
        sport = row.get("activity_type", "other")
        canonical = SPORT_MAP.get(sport, "other")
        dur = float(row.get("duration_seconds", 0) or 0)
        avg_pwr = float(row.get("avg_power", 0) or 0)
        avg_heart = float(row.get("avg_hr", 0) or 0)

        garmin_tss = row.get("tss")
        if pd.notna(garmin_tss) and float(garmin_tss) > 0:
            method = "garmin"
        elif canonical == "bike" and avg_pwr > 0:
            method = "power"
        elif canonical == "run":
            method = "pace"
        elif canonical == "swim":
            method = "swim"
        elif avg_heart > 0:
            method = "hr"
        else:
            method = "duration"

        tss_val = estimate_tss(row, ftp, lthr, css, run_threshold_pace)
        tss_values.append(tss_val)
        methods.append(method)

    df["tss_estimated"] = tss_values
    df["tss_method"] = methods
    df["sport_canonical"] = df["activity_type"].map(SPORT_MAP).fillna("other")

    return df


# ---------------------------------------------------------------------------
# Quick test / summary
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    csv_path = Path(__file__).parent.parent / "data" / "garmin" / "activities.csv"
    if not csv_path.exists():
        print(f"Activities CSV not found at {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} activities")
    print()

    print("Thresholds:")
    print(f"  FTP (bike):      {DEFAULT_FTP}W")
    print(f"  Run threshold:   {int(DEFAULT_RUN_THRESHOLD_PACE//60)}:{int(DEFAULT_RUN_THRESHOLD_PACE%60):02d}/mile")
    print(f"  LTHR:            {DEFAULT_LTHR} bpm")
    print(f"  CSS (swim):      {DEFAULT_SWIM_CSS} m/s")
    print()

    df = estimate_all_tss(df)

    print("TSS estimation method breakdown:")
    print(df["tss_method"].value_counts().to_string())
    print()

    print("TSS by sport (canonical):")
    summary = df.groupby("sport_canonical").agg(
        count=("tss_estimated", "count"),
        avg_tss=("tss_estimated", "mean"),
        total_tss=("tss_estimated", "sum"),
    ).round(1)
    print(summary.to_string())
    print()

    print(f"\nTotal estimated TSS across all activities: {df['tss_estimated'].sum():.0f}")
    print(f"Average TSS per activity: {df['tss_estimated'].mean():.1f}")

    # Save
    out_path = csv_path.parent / "activities_with_tss.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")
