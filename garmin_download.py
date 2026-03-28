#!/usr/bin/env python3
"""
Garmin Connect data downloader for TriML project.

Downloads ~15 months of training and health data from Garmin Connect and
saves each data type as a CSV to data/garmin/.

Usage:
    python3 garmin_download.py
    python3 garmin_download.py --start 2025-01-01 --end 2026-03-26
"""

import argparse
import csv
import getpass
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def date_range(start: date, end: date):
    """Yield each date from start through end (inclusive)."""
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def safe_get(d, *keys, default=None):
    """Safely traverse nested dicts/lists."""
    val = d
    for key in keys:
        if val is None:
            return default
        if isinstance(val, dict):
            val = val.get(key)
        elif isinstance(val, list) and isinstance(key, int):
            try:
                val = val[key]
            except IndexError:
                return default
        else:
            return default
    return val if val is not None else default


def print_progress(current: int, total: int, label: str = ""):
    pct = int(100 * current / total) if total else 0
    bar_len = 40
    filled = int(bar_len * current / total) if total else 0
    bar = "#" * filled + "-" * (bar_len - filled)
    print(f"\r[{bar}] {current}/{total} ({pct}%) {label}   ", end="", flush=True)


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

TOKEN_DIR = Path.home() / ".garmin_tokens"


def load_client_with_tokens():
    """
    Attempt to resume a saved garth session without re-entering credentials.
    Returns a logged-in Garmin client, or None on failure.
    """
    try:
        import garth
        from garminconnect import Garmin

        if not TOKEN_DIR.exists():
            return None

        garth.resume(str(TOKEN_DIR))
        client = Garmin()
        client.garth = garth
        # Verify session is still valid with a lightweight call
        client.get_full_name()
        print(f"Resumed session from {TOKEN_DIR}")
        return client
    except Exception as exc:
        print(f"Could not resume saved session ({exc}); will re-authenticate.")
        return None


def login_with_credentials():
    """Prompt for credentials, log in with retry/backoff, save tokens, return client."""
    from garminconnect import Garmin
    import garth

    print("\nGarmin Connect login")
    print("--------------------")
    email = input("Email: ").strip()
    password = getpass.getpass("Password: ")

    client = Garmin(email, password)

    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            print(f"  Login attempt {attempt}/{max_retries}...")
            client.login()
            print("  Login successful!")
            break
        except Exception as exc:
            if "429" in str(exc) and attempt < max_retries:
                wait = 60 * attempt  # 60s, 120s, 180s, 240s
                print(f"  Rate limited (429). Waiting {wait}s before retry...")
                time.sleep(wait)
            else:
                raise

    ensure_dir(TOKEN_DIR)
    # garth saves OAuth2 tokens – no plaintext password stored
    garth.save(str(TOKEN_DIR))
    print(f"Session tokens saved to {TOKEN_DIR}")
    return client


def get_authenticated_client():
    """Return an authenticated Garmin client, using cached tokens if available."""
    client = load_client_with_tokens()
    if client is None:
        client = login_with_credentials()
    return client


# ---------------------------------------------------------------------------
# Download helpers – one per data type
# ---------------------------------------------------------------------------

def download_daily_summary(client, dates, out_path: Path):
    """Download daily summary stats."""
    fieldnames = [
        "date", "resting_hr", "min_hr", "max_hr",
        "avg_stress", "max_stress",
        "body_battery_high", "body_battery_low",
        "steps", "floors_climbed",
        "active_calories", "total_calories",
    ]
    rows = []
    total = len(dates)
    print(f"\nDownloading daily summaries ({total} days)...")

    for i, d in enumerate(dates):
        print_progress(i + 1, total, str(d))
        date_str = d.isoformat()
        try:
            data = client.get_stats(date_str)
            row = {
                "date": date_str,
                "resting_hr": safe_get(data, "restingHeartRate"),
                "min_hr": safe_get(data, "minHeartRate"),
                "max_hr": safe_get(data, "maxHeartRate"),
                "avg_stress": safe_get(data, "averageStressLevel"),
                "max_stress": safe_get(data, "maxStressLevel"),
                "body_battery_high": safe_get(data, "bodyBatteryHighestValue"),
                "body_battery_low": safe_get(data, "bodyBatteryLowestValue"),
                "steps": safe_get(data, "totalSteps"),
                "floors_climbed": safe_get(data, "floorsAscended"),
                "active_calories": safe_get(data, "activeKilocalories"),
                "total_calories": safe_get(data, "totalKilocalories"),
            }
            rows.append(row)
        except Exception as exc:
            rows.append({"date": date_str, **{k: None for k in fieldnames if k != "date"}})
        time.sleep(0.5)

    print()  # newline after progress bar
    _write_csv(out_path, fieldnames, rows)
    return len(rows)


def download_sleep(client, dates, out_path: Path):
    """Download sleep data."""
    fieldnames = [
        "date", "sleep_start", "sleep_end",
        "total_sleep_seconds", "deep_sleep_seconds",
        "light_sleep_seconds", "rem_sleep_seconds",
        "awake_seconds", "sleep_score",
    ]
    rows = []
    total = len(dates)
    print(f"\nDownloading sleep data ({total} days)...")

    for i, d in enumerate(dates):
        print_progress(i + 1, total, str(d))
        date_str = d.isoformat()
        try:
            data = client.get_sleep_data(date_str)
            daily = safe_get(data, "dailySleepDTO") or {}
            row = {
                "date": date_str,
                "sleep_start": safe_get(daily, "sleepStartTimestampGMT"),
                "sleep_end": safe_get(daily, "sleepEndTimestampGMT"),
                "total_sleep_seconds": safe_get(daily, "sleepTimeSeconds"),
                "deep_sleep_seconds": safe_get(daily, "deepSleepSeconds"),
                "light_sleep_seconds": safe_get(daily, "lightSleepSeconds"),
                "rem_sleep_seconds": safe_get(daily, "remSleepSeconds"),
                "awake_seconds": safe_get(daily, "awakeSleepSeconds"),
                "sleep_score": safe_get(data, "sleepScores", "overall", "value")
                               or safe_get(daily, "sleepScores", "overall"),
            }
            rows.append(row)
        except Exception:
            rows.append({"date": date_str, **{k: None for k in fieldnames if k != "date"}})
        time.sleep(0.5)

    print()
    _write_csv(out_path, fieldnames, rows)
    return len(rows)


def download_hrv(client, dates, out_path: Path):
    """Download HRV data."""
    fieldnames = [
        "date", "hrv_weekly_avg", "hrv_last_night",
        "hrv_status", "baseline_low", "baseline_high",
    ]
    rows = []
    total = len(dates)
    print(f"\nDownloading HRV data ({total} days)...")

    for i, d in enumerate(dates):
        print_progress(i + 1, total, str(d))
        date_str = d.isoformat()
        try:
            data = client.get_hrv_data(date_str)
            summary = safe_get(data, "hrvSummary") or {}
            row = {
                "date": date_str,
                "hrv_weekly_avg": safe_get(summary, "weeklyAvg"),
                "hrv_last_night": safe_get(summary, "lastNight"),
                "hrv_status": safe_get(summary, "status"),
                "baseline_low": safe_get(summary, "baseline", "lowUpper")
                                or safe_get(data, "startTimestampGMT"),  # fallback field probe
                "baseline_high": safe_get(summary, "baseline", "balancedLow"),
            }
            # Re-map baseline fields properly
            baseline = safe_get(summary, "baseline") or {}
            row["baseline_low"] = baseline.get("lowUpper") or baseline.get("balanced_low")
            row["baseline_high"] = baseline.get("balancedHigh") or baseline.get("balanced_high")
            rows.append(row)
        except Exception:
            rows.append({"date": date_str, **{k: None for k in fieldnames if k != "date"}})
        time.sleep(0.5)

    print()
    _write_csv(out_path, fieldnames, rows)
    return len(rows)


def download_activities(client, start_date: date, end_date: date, out_path: Path):
    """Download all activities in the date range in one batch call."""
    fieldnames = [
        "date", "activity_type", "sport", "duration_seconds",
        "distance_meters", "avg_hr", "max_hr", "calories",
        "avg_power", "tss", "training_effect_aerobic",
        "training_effect_anaerobic", "activity_name",
    ]
    print(f"\nDownloading activities ({start_date} → {end_date})...")
    rows = []
    try:
        activities = client.get_activities_by_date(
            start_date.isoformat(), end_date.isoformat()
        )
        print(f"  Found {len(activities)} activities.")
        for act in activities:
            start_ts = safe_get(act, "startTimeLocal") or safe_get(act, "startTimeGMT", "")
            act_date = start_ts[:10] if start_ts else None
            row = {
                "date": act_date,
                "activity_type": safe_get(act, "activityType", "typeKey"),
                "sport": safe_get(act, "activityType", "typeKey"),
                "duration_seconds": safe_get(act, "duration"),
                "distance_meters": safe_get(act, "distance"),
                "avg_hr": safe_get(act, "averageHR"),
                "max_hr": safe_get(act, "maxHR"),
                "calories": safe_get(act, "calories"),
                "avg_power": safe_get(act, "avgPower"),
                "tss": safe_get(act, "tss"),
                "training_effect_aerobic": safe_get(act, "aerobicTrainingEffect"),
                "training_effect_anaerobic": safe_get(act, "anaerobicTrainingEffect"),
                "activity_name": safe_get(act, "activityName"),
            }
            rows.append(row)
    except Exception as exc:
        print(f"  Warning: could not retrieve activities: {exc}")

    _write_csv(out_path, fieldnames, rows)
    return len(rows)


def download_stress(client, dates, out_path: Path):
    """Download daily stress data."""
    fieldnames = [
        "date", "avg_stress",
        "high_stress_duration", "medium_stress_duration",
        "low_stress_duration", "rest_stress_duration",
    ]
    rows = []
    total = len(dates)
    print(f"\nDownloading stress data ({total} days)...")

    for i, d in enumerate(dates):
        print_progress(i + 1, total, str(d))
        date_str = d.isoformat()
        try:
            data = client.get_stress_data(date_str)
            row = {
                "date": date_str,
                "avg_stress": safe_get(data, "overallStressLevel"),
                "high_stress_duration": safe_get(data, "highStressDuration"),
                "medium_stress_duration": safe_get(data, "mediumStressDuration"),
                "low_stress_duration": safe_get(data, "lowStressDuration"),
                "rest_stress_duration": safe_get(data, "restStressDuration"),
            }
            rows.append(row)
        except Exception:
            rows.append({"date": date_str, **{k: None for k in fieldnames if k != "date"}})
        time.sleep(0.5)

    print()
    _write_csv(out_path, fieldnames, rows)
    return len(rows)


def download_body_battery(client, dates, out_path: Path):
    """Download daily body battery data."""
    fieldnames = [
        "date", "morning_value", "evening_value", "high", "low",
    ]
    rows = []
    total = len(dates)
    print(f"\nDownloading body battery data ({total} days)...")

    for i, d in enumerate(dates):
        print_progress(i + 1, total, str(d))
        date_str = d.isoformat()
        try:
            # get_body_battery returns a list of readings for the day
            data = client.get_body_battery(date_str)

            # data can be a list of dicts with 'charged'/'drained' values,
            # or a list of [timestamp, value] pairs depending on the endpoint.
            values = []
            if isinstance(data, list):
                for entry in data:
                    if isinstance(entry, dict):
                        v = entry.get("bodyBatteryLevel") or entry.get("value")
                    elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                        v = entry[1]
                    else:
                        v = None
                    if v is not None:
                        values.append(int(v))

            morning = values[0] if values else None
            evening = values[-1] if values else None
            high = max(values) if values else None
            low = min(values) if values else None

            rows.append({
                "date": date_str,
                "morning_value": morning,
                "evening_value": evening,
                "high": high,
                "low": low,
            })
        except Exception:
            rows.append({"date": date_str, **{k: None for k in fieldnames if k != "date"}})
        time.sleep(0.5)

    print()
    _write_csv(out_path, fieldnames, rows)
    return len(rows)


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def _write_csv(path: Path, fieldnames: list, rows: list):
    ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {len(rows)} rows → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Garmin Connect data to CSV files."
    )
    parser.add_argument(
        "--start", default="2025-01-01",
        help="Start date (YYYY-MM-DD), default: 2025-01-01"
    )
    parser.add_argument(
        "--end", default=date.today().isoformat(),
        help=f"End date (YYYY-MM-DD), default: today ({date.today().isoformat()})"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        start_date = date.fromisoformat(args.start)
        end_date = date.fromisoformat(args.end)
    except ValueError as exc:
        print(f"Invalid date format: {exc}")
        sys.exit(1)

    if start_date > end_date:
        print("Error: --start must be before --end")
        sys.exit(1)

    print(f"Date range: {start_date} → {end_date}")
    total_days = (end_date - start_date).days + 1
    print(f"Total days: {total_days}")

    # Output directory
    out_dir = Path(__file__).parent / "data" / "garmin"
    ensure_dir(out_dir)
    print(f"Output directory: {out_dir}")

    # Authenticate
    try:
        client = get_authenticated_client()
    except Exception as exc:
        print(f"\nAuthentication failed: {exc}")
        sys.exit(1)

    dates = list(date_range(start_date, end_date))

    # ---- Download each data type ----
    counts = {}

    counts["daily_summary"] = download_daily_summary(
        client, dates, out_dir / "daily_summary.csv"
    )

    counts["sleep"] = download_sleep(
        client, dates, out_dir / "sleep.csv"
    )

    counts["hrv"] = download_hrv(
        client, dates, out_dir / "hrv.csv"
    )

    counts["activities"] = download_activities(
        client, start_date, end_date, out_dir / "activities.csv"
    )

    counts["stress"] = download_stress(
        client, dates, out_dir / "stress.csv"
    )

    counts["body_battery"] = download_body_battery(
        client, dates, out_dir / "body_battery.csv"
    )

    # ---- Summary ----
    print("\n" + "=" * 50)
    print("Download complete — row counts:")
    print("=" * 50)
    for name, count in counts.items():
        print(f"  {name:<20} {count:>6} rows")
    print("=" * 50)


if __name__ == "__main__":
    main()
