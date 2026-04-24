#!/usr/bin/env python3
# FutureSteps: tp_download.py
"""
TrainingPeaks API data downloader for TriML project.

Downloads planned and completed workouts via OAuth2.
Saves to data/trainingpeaks/.

Usage:
    python3 tp_download.py
    python3 tp_download.py --start 2025-01-01 --end 2026-03-31
"""

import argparse
import csv
import getpass
import http.server
import json
import sys
import time
import urllib.parse
import webbrowser
from datetime import date
from pathlib import Path

import requests

# TrainingPeaks API config
AUTH_URL = "https://oauth.trainingpeaks.com/OAuth/Authorize"
TOKEN_URL = "https://oauth.trainingpeaks.com/oauth/token"
API_BASE = "https://api.trainingpeaks.com"
REDIRECT_URI = "http://localhost:8000/callback"
SCOPES = "workouts athlete"

TOKEN_DIR = Path.home() / ".tp_tokens"
TOKEN_FILE = TOKEN_DIR / "tokens.json"


# Helpers

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_tokens(tokens):
    ensure_dir(TOKEN_DIR)
    with open(TOKEN_FILE, "w") as f:
        json.dump(tokens, f)


def load_tokens():
    if TOKEN_FILE.exists():
        with open(TOKEN_FILE) as f:
            return json.load(f)
    return None


def _write_csv(path, fieldnames, rows):
    ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {len(rows)} rows -> {path}")


# OAuth2 flow

def get_auth_code(client_id):
    """Open browser for TrainingPeaks login and capture auth code."""
    params = {
        "client_id": client_id,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": SCOPES,
    }
    auth_url = f"{AUTH_URL}?{urllib.parse.urlencode(params)}"

    auth_code = None

    class CallbackHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            nonlocal auth_code
            query = urllib.parse.urlparse(self.path).query
            qs = urllib.parse.parse_qs(query)
            auth_code = qs.get("code", [None])[0]

            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<h1>Success! You can close this tab and return to the terminal.</h1>"
            )

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("localhost", 8000), CallbackHandler)
    print("\nOpening browser for TrainingPeaks login...")
    webbrowser.open(auth_url)
    print("Waiting for callback (log in via the browser)...")
    server.handle_request()
    server.server_close()

    if not auth_code:
        print("ERROR: No authorization code received.")
        sys.exit(1)

    print("Authorization code received!")
    return auth_code


def exchange_code(client_id, client_secret, code):
    """Exchange authorization code for access + refresh tokens."""
    resp = requests.post(TOKEN_URL, data={
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": REDIRECT_URI,
        "client_id": client_id,
        "client_secret": client_secret,
    })
    resp.raise_for_status()
    tokens = resp.json()
    save_tokens(tokens)
    print("Tokens saved!")
    return tokens


def refresh_access_token(client_id, client_secret, refresh_token):
    """Refresh an expired access token."""
    resp = requests.post(TOKEN_URL, data={
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
        "client_secret": client_secret,
    })
    resp.raise_for_status()
    tokens = resp.json()
    save_tokens(tokens)
    return tokens


def authenticate():
    """Full auth flow: try saved tokens -> refresh -> full OAuth2."""
    print("\nTrainingPeaks Authentication")
    print("----------------------------")
    client_id = input("Client ID: ").strip()
    client_secret = getpass.getpass("Client Secret: ")

    tokens = load_tokens()

    if tokens and tokens.get("refresh_token"):
        try:
            print("Refreshing saved token...")
            tokens = refresh_access_token(
                client_id, client_secret, tokens["refresh_token"]
            )
            print("Token refreshed successfully!")
            return tokens["access_token"], client_id, client_secret
        except Exception as exc:
            print(f"Refresh failed ({exc}), starting fresh OAuth2 flow...")

    code = get_auth_code(client_id)
    tokens = exchange_code(client_id, client_secret, code)
    return tokens["access_token"], client_id, client_secret


# API helpers

def api_get(endpoint, access_token, params=None):
    """Authenticated GET with retry on 429."""
    url = f"{API_BASE}{endpoint}"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
    }

    for attempt in range(5):
        resp = requests.get(url, headers=headers, params=params)
        if resp.status_code == 429:
            wait = 60 * (attempt + 1)
            print(f"\n  Rate limited (429). Waiting {wait}s...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()

    raise Exception("Max retries exceeded on 429")


def get_athlete_id(access_token):
    """Get the authenticated athlete's ID."""
    data = api_get("/v1/athlete/profile", access_token)
    athlete_id = data.get("Id") or data.get("id")
    if not athlete_id:
        # Try alternative endpoint
        data = api_get("/v1/athlete", access_token)
        athlete_id = data.get("Id") or data.get("id")
    print(f"  Athlete ID: {athlete_id}")
    return athlete_id


# Download functions

def download_completed_workouts(access_token, athlete_id, start_date, end_date, out_path):
    """Download completed (actual) workouts."""
    fieldnames = [
        "date", "workout_type", "sport", "title",
        "duration_seconds", "distance_meters",
        "tss", "planned_tss", "intensity_factor", "normalized_power",
        "avg_hr", "max_hr", "avg_power", "max_power",
        "calories", "elevation_gain",
    ]

    print(f"\nDownloading completed workouts ({start_date} -> {end_date})...")
    data = api_get(
        f"/v1/athlete/{athlete_id}/workouts",
        access_token,
        params={"startDate": start_date, "endDate": end_date},
    )

    if not isinstance(data, list):
        data = data.get("workouts", []) if isinstance(data, dict) else []

    print(f"  Found {len(data)} workouts.")

    rows = []
    for w in data:
        # Extract date from StartTime or WorkoutDay
        workout_date = None
        for key in ["WorkoutDay", "StartTime", "startTime", "workoutDay"]:
            val = w.get(key)
            if val and isinstance(val, str):
                workout_date = val[:10]
                break

        # Map TP workout types to our sport names
        wtype = (
            w.get("WorkoutType") or w.get("workoutType") or
            w.get("Type") or w.get("type") or ""
        ).lower()

        sport = "other"
        if "bike" in wtype or "cycl" in wtype or "ride" in wtype:
            sport = "cycling"
        elif "run" in wtype or "jog" in wtype:
            sport = "running"
        elif "swim" in wtype:
            sport = "swimming"
        elif "strength" in wtype or "weight" in wtype:
            sport = "strength"
        elif "yoga" in wtype:
            sport = "yoga"
        elif "row" in wtype:
            sport = "rowing"

        def g(key):
            """Case-insensitive field getter."""
            val = w.get(key)
            if val is not None:
                return val
            # Try PascalCase
            pascal = key[0].upper() + key[1:]
            return w.get(pascal)

        rows.append({
            "date": workout_date,
            "workout_type": wtype,
            "sport": sport,
            "title": g("title") or g("Title") or g("workoutName"),
            "duration_seconds": g("totalTimePlanned") or g("duration") or g("TimeTotalInSeconds"),
            "distance_meters": g("distancePlanned") or g("distance") or g("DistanceInMeters"),
            "tss": g("tpiTss") or g("tss") or g("TSS") or g("TssActual"),
            "planned_tss": g("tpiTssPlanned") or g("TssPlanned") or g("tssPlanned"),
            "intensity_factor": g("intensityFactor") or g("IntensityFactor") or g("IF"),
            "normalized_power": g("normalizedPower") or g("NormalizedPower"),
            "avg_hr": g("heartRateAverage") or g("HeartRateAverage"),
            "max_hr": g("heartRateMaximum") or g("HeartRateMaximum"),
            "avg_power": g("powerAverage") or g("PowerAverage"),
            "max_power": g("powerMaximum") or g("PowerMaximum"),
            "calories": g("calories") or g("Calories") or g("CaloriesSpent"),
            "elevation_gain": g("elevationGain") or g("ElevationGain"),
        })

    _write_csv(out_path, fieldnames, rows)
    return len(rows)


def download_planned_workouts(access_token, athlete_id, start_date, end_date, out_path):
    """Download planned (coach-prescribed) workouts."""
    fieldnames = [
        "date", "workout_type", "sport", "title", "description",
        "planned_duration_seconds", "planned_distance_meters",
        "planned_tss", "planned_if",
    ]

    print(f"\nDownloading planned workouts ({start_date} -> {end_date})...")

    # TP separates planned vs completed in some API versions
    # Try the planned endpoint first, fall back to filtering from workouts
    try:
        data = api_get(
            f"/v1/athlete/{athlete_id}/workouts",
            access_token,
            params={"startDate": start_date, "endDate": end_date},
        )
    except Exception as exc:
        print(f"  Could not fetch planned workouts: {exc}")
        _write_csv(out_path, fieldnames, [])
        return 0

    if not isinstance(data, list):
        data = data.get("workouts", []) if isinstance(data, dict) else []

    rows = []
    for w in data:
        workout_date = None
        for key in ["WorkoutDay", "StartTime", "startTime", "workoutDay"]:
            val = w.get(key)
            if val and isinstance(val, str):
                workout_date = val[:10]
                break

        wtype = (
            w.get("WorkoutType") or w.get("workoutType") or
            w.get("Type") or w.get("type") or ""
        ).lower()

        sport = "other"
        if "bike" in wtype or "cycl" in wtype or "ride" in wtype:
            sport = "cycling"
        elif "run" in wtype or "jog" in wtype:
            sport = "running"
        elif "swim" in wtype:
            sport = "swimming"
        elif "strength" in wtype or "weight" in wtype:
            sport = "strength"

        def g(key):
            val = w.get(key)
            if val is not None:
                return val
            pascal = key[0].upper() + key[1:]
            return w.get(pascal)

        planned_tss = g("TssPlanned") or g("tssPlanned") or g("tpiTssPlanned")
        planned_dur = g("TimeTotalInSecondsPlanned") or g("durationPlanned")
        planned_dist = g("DistancePlannedInMeters") or g("distancePlanned")
        planned_if = g("IntensityFactorPlanned") or g("ifPlanned")

        # Only include if there's planned data
        if planned_tss or planned_dur:
            rows.append({
                "date": workout_date,
                "workout_type": wtype,
                "sport": sport,
                "title": g("title") or g("Title") or g("workoutName"),
                "description": g("description") or g("Description") or "",
                "planned_duration_seconds": planned_dur,
                "planned_distance_meters": planned_dist,
                "planned_tss": planned_tss,
                "planned_if": planned_if,
            })

    _write_csv(out_path, fieldnames, rows)
    return len(rows)


# Main

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download TrainingPeaks data to CSV files."
    )
    parser.add_argument(
        "--start", default="2025-01-01",
        help="Start date (YYYY-MM-DD), default: 2025-01-01",
    )
    parser.add_argument(
        "--end", default=date.today().isoformat(),
        help=f"End date (YYYY-MM-DD), default: today ({date.today().isoformat()})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("TrainingPeaks Data Downloader")
    print(f"Date range: {args.start} -> {args.end}")
    print("=" * 50)

    access_token, client_id, client_secret = authenticate()

    out_dir = Path(__file__).parent / "data" / "trainingpeaks"
    ensure_dir(out_dir)

    # Get athlete ID
    print("\nFetching athlete profile...")
    athlete_id = get_athlete_id(access_token)

    counts = {}

    counts["completed"] = download_completed_workouts(
        access_token, athlete_id,
        args.start, args.end,
        out_dir / "completed_workouts.csv",
    )

    counts["planned"] = download_planned_workouts(
        access_token, athlete_id,
        args.start, args.end,
        out_dir / "planned_workouts.csv",
    )

    # Summary
    print("\n" + "=" * 50)
    print("Download complete — row counts:")
    print("=" * 50)
    for name, count in counts.items():
        print(f"  {name:<20} {count:>6} rows")
    print("=" * 50)
    print(f"\nFiles saved to: {out_dir}")


if __name__ == "__main__":
    main()
