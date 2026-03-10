"""
Data loading and parsing for the TriML project.

Handles the three raw CSVs:
  - athletes.csv       : 1,000 static athlete profiles
  - daily_data.csv     : 366,000 daily wearable readings (1,000 athletes × 366 days)
  - activity_data.csv  : 384,153 training session records

All string-encoded columns (hrv_range, hr_zones, power_zones) are parsed
into proper numeric columns here so downstream code never has to deal with them.
"""

import ast
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

# Default path: CSVs live in the project root
_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _ROOT  # athletes.csv / daily_data.csv / activity_data.csv are in root

# Public Zenodo download URLs (record 15401061)
_ZENODO_URLS = {
    "athletes.csv": "https://zenodo.org/api/records/15401061/files/athletes.csv/content",
    "daily_data.csv": "https://zenodo.org/api/records/15401061/files/daily_data.csv/content",
    "activity_data.csv": "https://zenodo.org/api/records/15401061/files/activity_data.csv/content",
}

_FILE_SIZES = {
    "athletes.csv": 465_161,
    "daily_data.csv": 71_186_033,
    "activity_data.csv": 115_428_530,
}


def ensure_data(data_dir: Path | None = None, progress_callback=None) -> Path:
    """
    Make sure all three CSV files exist in data_dir (defaults to project root).
    If any are missing, download them from Zenodo.

    Args:
        data_dir: Directory to store/find CSVs. Defaults to project root.
        progress_callback: Optional callable(filename, bytes_downloaded, total_bytes)
                           for progress reporting (used by Streamlit spinner).

    Returns:
        The resolved data_dir Path.
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR

    for filename, url in _ZENODO_URLS.items():
        dest = data_dir / filename
        if dest.exists() and dest.stat().st_size > _FILE_SIZES[filename] * 0.9:
            continue  # already present and looks complete

        total = _FILE_SIZES[filename]
        downloaded = 0

        def _hook(block_num, block_size, file_size):
            nonlocal downloaded
            downloaded = min(block_num * block_size, total)
            if progress_callback:
                progress_callback(filename, downloaded, total)

        urllib.request.urlretrieve(url, dest, reporthook=_hook)

    return data_dir


# ---------------------------------------------------------------------------
# Internal parsers for string-encoded columns
# ---------------------------------------------------------------------------

def _parse_hrv_range(s: str) -> tuple[float, float]:
    """
    Parse hrv_range strings like '(np.float64(82.9), np.float64(112.1))'
    into a (min, max) float tuple.
    """
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", str(s))
    return float(nums[0]), float(nums[1])


def _parse_hr_zones_athlete(s: str) -> dict:
    """
    Parse athletes.csv hr_zones strings that contain np.float64() wrappers.
    Returns dict like {'Z1': (lo, hi), 'Z2': (lo, hi), ...} with plain floats.
    """
    cleaned = re.sub(r"np\.float64\(([^)]+)\)", r"\1", str(s))
    return ast.literal_eval(cleaned)


def _parse_zone_dict(s) -> dict | None:
    """
    Parse activity_data hr_zones / power_zones — clean dicts, no wrappers.
    Returns None for missing values.
    """
    if pd.isna(s):
        return None
    return ast.literal_eval(str(s))


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_athletes(path: str | Path | None = None) -> pd.DataFrame:
    """
    Load athletes.csv and expand string-encoded columns into numeric ones.

    Added columns (replacing originals):
      hrv_min, hrv_max           – from hrv_range
      hr_zone{1..6}_lo/hi        – from hr_zones (12 boundary columns)

    Returns:
        DataFrame, 1,000 rows × ~34 columns, no nulls.
    """
    path = Path(path) if path else DATA_DIR / "athletes.csv"
    df = pd.read_csv(path)

    # --- hrv_range → hrv_min, hrv_max ---
    parsed = df["hrv_range"].apply(_parse_hrv_range)
    df["hrv_min"] = parsed.apply(lambda t: t[0])
    df["hrv_max"] = parsed.apply(lambda t: t[1])
    df.drop(columns=["hrv_range"], inplace=True)

    # --- hr_zones → 12 boundary columns ---
    parsed_zones = df["hr_zones"].apply(_parse_hr_zones_athlete)
    for z in range(1, 7):
        key = f"Z{z}"
        df[f"hr_zone{z}_lo"] = parsed_zones.apply(lambda d, k=key: float(d[k][0]))
        df[f"hr_zone{z}_hi"] = parsed_zones.apply(lambda d, k=key: float(d[k][1]))
    df.drop(columns=["hr_zones"], inplace=True)

    return df


def load_daily(path: str | Path | None = None) -> pd.DataFrame:
    """
    Load daily_data.csv. Parses date column; sorts by (athlete_id, date).

    Returns:
        DataFrame, 366,000 rows × 15 columns, no nulls.
    """
    path = Path(path) if path else DATA_DIR / "daily_data.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df.sort_values(["athlete_id", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_activities(path: str | Path | None = None) -> pd.DataFrame:
    """
    Load activity_data.csv and expand hr_zones / power_zones into numeric columns.

    Added columns (replacing originals):
      hr_z{1..6}_pct     – percent time in each HR zone
      pwr_z{1..7}_pct    – percent time in each power zone (0 for non-cycling)

    Returns:
        DataFrame, 384,153 rows × ~33 columns.
    """
    path = Path(path) if path else DATA_DIR / "activity_data.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df.sort_values(["athlete_id", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # --- hr_zones → 6 percentage columns ---
    parsed_hr = df["hr_zones"].apply(_parse_zone_dict)
    for z in range(1, 7):
        key = f"Z{z}"
        df[f"hr_z{z}_pct"] = parsed_hr.apply(
            lambda d, k=key: float(d[k]) if d is not None else 0.0
        )
    df.drop(columns=["hr_zones"], inplace=True)

    # --- power_zones → 7 percentage columns (0 for non-cycling) ---
    parsed_pwr = df["power_zones"].apply(_parse_zone_dict)
    for z in range(1, 8):
        key = f"Z{z}"
        df[f"pwr_z{z}_pct"] = parsed_pwr.apply(
            lambda d, k=key: float(d[k]) if d is not None else 0.0
        )
    df.drop(columns=["power_zones"], inplace=True)

    return df


def aggregate_activities(df_act: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse activity_data to one row per (athlete_id, date).

    Aggregations:
      Sum:   tss, duration_minutes, work_kilojoules
      Mean:  intensity_factor, avg_hr, training_effect_aerobic,
             training_effect_anaerobic, hr_z*_pct
      Count per sport: n_bike, n_run, n_swim, n_strength
      Sports list:     sports_of_day (e.g. "bike,run")
      Dominant sport:  dominant_sport (highest-TSS sport that day)

    Returns:
        DataFrame, one row per (athlete_id, date).
    """
    zone_cols = [f"hr_z{z}_pct" for z in range(1, 7)] + [
        f"pwr_z{z}_pct" for z in range(1, 8)
    ]

    # Per-sport session counts
    sport_dummies = pd.get_dummies(df_act["sport"], prefix="n")
    df_act = pd.concat([df_act, sport_dummies], axis=1)
    sport_count_cols = [c for c in df_act.columns if c.startswith("n_")]

    # Dominant sport per day (sport with most TSS; ties broken alphabetically)
    def _dominant_sport(sub):
        return sub.groupby("sport")["tss"].sum().idxmax()

    dominant = (
        df_act.groupby(["athlete_id", "date"])
        .apply(_dominant_sport, include_groups=False)
        .rename("dominant_sport")
        .reset_index()
    )

    # Sports list per day
    sports_list = (
        df_act.groupby(["athlete_id", "date"])["sport"]
        .apply(lambda x: ",".join(sorted(x.unique())))
        .rename("sports_of_day")
        .reset_index()
    )

    # Main aggregation
    agg_dict = {
        "tss": "sum",
        "duration_minutes": "sum",
        "work_kilojoules": "sum",
        "intensity_factor": "mean",
        "avg_hr": "mean",
        "training_effect_aerobic": "mean",
        "training_effect_anaerobic": "mean",
    }
    for col in zone_cols:
        agg_dict[col] = "mean"
    for col in sport_count_cols:
        agg_dict[col] = "sum"

    agg = df_act.groupby(["athlete_id", "date"]).agg(agg_dict).reset_index()

    # Rename sport count columns: n_bike, n_run, n_swim, n_strength
    rename_map = {c: c.replace("n_", "n_") for c in sport_count_cols}
    # Ensure canonical column names regardless of get_dummies prefix
    for sport in ["bike", "run", "swim", "strength"]:
        old = f"n_{sport}"
        if old not in agg.columns:
            agg[old] = 0

    agg = agg.merge(dominant, on=["athlete_id", "date"], how="left")
    agg = agg.merge(sports_list, on=["athlete_id", "date"], how="left")

    return agg


def build_merged(
    df_daily: pd.DataFrame,
    df_act_agg: pd.DataFrame,
    df_athletes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join all three tables into one analysis-ready DataFrame.

      daily_data (366,000 rows)
        ← activity aggregate on (athlete_id, date)   [rest days get 0]
        ← athlete static features on athlete_id

    Returns:
        DataFrame, 366,000 rows × ~70 columns, no nulls in core columns.
    """
    # Merge activity aggregates onto daily rows (rest days get NaN → fill 0)
    df = df_daily.merge(df_act_agg, on=["athlete_id", "date"], how="left")

    activity_numeric_cols = [
        c for c in df_act_agg.columns
        if c not in ("athlete_id", "date", "dominant_sport", "sports_of_day")
    ]
    df[activity_numeric_cols] = df[activity_numeric_cols].fillna(0)
    df["dominant_sport"] = df["dominant_sport"].fillna("rest")
    df["sports_of_day"] = df["sports_of_day"].fillna("")

    # Rename athlete columns that collide with daily_data columns before merging
    # (athletes has baseline values; daily has actual daily measurements)
    df_athletes = df_athletes.rename(columns={
        "resting_hr": "baseline_rhr",
        "sleep_quality": "baseline_sleep_quality",
    })

    # Merge static athlete profile
    df = df.merge(df_athletes, on="athlete_id", how="left")

    df.sort_values(["athlete_id", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def find_demo_athlete(df_merged: pd.DataFrame, df_act: pd.DataFrame) -> str:
    """
    Return the athlete_id of the most-injured athlete who also has all four
    sport types in their activity log. Used as the default in the Streamlit app.
    """
    # Athletes with all 4 sports
    sports_per_athlete = df_act.groupby("athlete_id")["sport"].apply(set)
    full_sport_athletes = sports_per_athlete[
        sports_per_athlete.apply(lambda s: {"bike", "run", "swim", "strength"}.issubset(s))
    ].index

    # Among those, pick the one with most injury days
    injury_counts = (
        df_merged[df_merged["athlete_id"].isin(full_sport_athletes)]
        .groupby("athlete_id")["injury"]
        .sum()
    )
    return injury_counts.idxmax()
