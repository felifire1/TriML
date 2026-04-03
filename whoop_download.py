#!/usr/bin/env python3
"""
WHOOP API data downloader for TriML project.

Downloads recovery, sleep, workout, and cycle data via OAuth2.
Saves each data type as CSV to data/whoop/.

Usage:
    python3 whoop_download.py
    python3 whoop_download.py --start 2025-01-01 --end 2026-03-31
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

# ---------------------------------------------------------------------------
# WHOOP API config
# ---------------------------------------------------------------------------
AUTH_URL = "https://api.prod.whoop.com/oauth/oauth2/auth"
TOKEN_URL = "https://api.prod.whoop.com/oauth/oauth2/token"
API_BASE = "https://api.prod.whoop.com/developer"
REDIRECT_URI = "http://localhost:8000/callback"
SCOPES = "read:recovery read:sleep read:workout read:cycles read:profile"

TOKEN_DIR = Path.home() / ".whoop_tokens"
TOKEN_FILE = TOKEN_DIR / "tokens.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# OAuth2 flow
# ---------------------------------------------------------------------------

def get_auth_code(client_id):
    """Open browser for WHOOP login and capture the auth code via localhost."""
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
    print("\nOpening browser for WHOOP login...")
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
    print("\nWHOOP Authentication")
    print("--------------------")
    client_id = input("Client ID: ").strip()
    client_secret = getpass.getpass("Client Secret: ")

    tokens = load_tokens()

    # Try refreshing existing token first
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

    # Full OAuth2 flow
    code = get_auth_code(client_id)
    tokens = exchange_code(client_id, client_secret, code)
    return tokens["access_token"], client_id, client_secret


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def api_get(endpoint, access_token, params=None):
    """Authenticated GET with retry on 429."""
    url = f"{API_BASE}{endpoint}"
    headers = {"Authorization": f"Bearer {access_token}"}

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


def fetch_all_pages(endpoint, access_token, start_iso, end_iso):
    """Paginate through a WHOOP endpoint, collecting all records."""
    all_records = []
    next_token = None
    page = 0

    while True:
        page += 1
        params = {"limit": 25, "start": start_iso, "end": end_iso}
        if next_token:
            params["nextToken"] = next_token

        data = api_get(endpoint, access_token, params)
        records = data.get("records", [])
        all_records.extend(records)
        print(f"\r  Page {page}: {len(all_records)} records so far...", end="", flush=True)

        next_token = data.get("next_token")
        if not next_token or not records:
            break
        time.sleep(0.3)

    print()
    return all_records


# ---------------------------------------------------------------------------
# Extract date helper
# ---------------------------------------------------------------------------

def extract_date(record):
    """Pull a YYYY-MM-DD date from various WHOOP timestamp fields."""
    for key in ["created_at", "start", "updated_at"]:
        val = record.get(key)
        if val and isinstance(val, str) and len(val) >= 10:
            return val[:10]
    return None


# ---------------------------------------------------------------------------
# Download + CSV writers per data type
# ---------------------------------------------------------------------------

def download_recovery(records, out_path):
    fieldnames = [
        "date", "recovery_score", "resting_hr", "hrv_rmssd",
        "skin_temp", "spo2",
    ]
    rows = []
    for r in records:
        score = r.get("score") or {}
        rows.append({
            "date": extract_date(r),
            "recovery_score": score.get("recovery_score"),
            "resting_hr": score.get("resting_heart_rate"),
            "hrv_rmssd": score.get("hrv_rmssd_milli"),
            "skin_temp": score.get("skin_temp_celsius"),
            "spo2": score.get("spo2_percentage"),
        })
    _write_csv(out_path, fieldnames, rows)
    return len(rows)


def download_sleep(records, out_path):
    fieldnames = [
        "date", "total_sleep_ms", "light_sleep_ms", "rem_sleep_ms",
        "slow_wave_sleep_ms", "wake_ms", "sleep_performance",
        "sleep_consistency", "sleep_efficiency",
    ]
    rows = []
    for r in records:
        score = r.get("score") or {}
        stage = score.get("stage_summary") or {}
        rows.append({
            "date": extract_date(r),
            "total_sleep_ms": stage.get("total_in_bed_time_milli"),
            "light_sleep_ms": stage.get("total_light_sleep_time_milli"),
            "rem_sleep_ms": stage.get("total_rem_sleep_time_milli"),
            "slow_wave_sleep_ms": stage.get("total_slow_wave_sleep_time_milli"),
            "wake_ms": stage.get("total_wake_time_milli"),
            "sleep_performance": score.get("sleep_performance_percentage"),
            "sleep_consistency": score.get("sleep_consistency_percentage"),
            "sleep_efficiency": score.get("sleep_efficiency_percentage"),
        })
    _write_csv(out_path, fieldnames, rows)
    return len(rows)


def download_workouts(records, out_path):
    fieldnames = [
        "date", "sport_id", "sport_name", "strain", "avg_hr", "max_hr",
        "kilojoules", "duration_ms", "distance_meters",
    ]

    # WHOOP sport ID -> name mapping (common ones)
    SPORT_MAP = {
        0: "running", 1: "cycling", 16: "swimming", 43: "strength",
        44: "yoga", 52: "hiking", 63: "triathlon", 71: "rowing",
        -1: "other",
    }

    rows = []
    for r in records:
        score = r.get("score") or {}
        sport_id = r.get("sport_id", -1)
        rows.append({
            "date": extract_date(r),
            "sport_id": sport_id,
            "sport_name": SPORT_MAP.get(sport_id, f"sport_{sport_id}"),
            "strain": score.get("strain"),
            "avg_hr": score.get("average_heart_rate"),
            "max_hr": score.get("max_heart_rate"),
            "kilojoules": score.get("kilojoule"),
            "duration_ms": score.get("total_time_milli"),
            "distance_meters": score.get("distance_meter"),
        })
    _write_csv(out_path, fieldnames, rows)
    return len(rows)


def download_cycles(records, out_path):
    fieldnames = ["date", "strain", "kilojoules", "avg_hr", "max_hr"]
    rows = []
    for r in records:
        score = r.get("score") or {}
        rows.append({
            "date": extract_date(r),
            "strain": score.get("strain"),
            "kilojoules": score.get("kilojoule"),
            "avg_hr": score.get("average_heart_rate"),
            "max_hr": score.get("max_heart_rate"),
        })
    _write_csv(out_path, fieldnames, rows)
    return len(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download WHOOP data to CSV files."
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
    print("WHOOP Data Downloader")
    print(f"Date range: {args.start} -> {args.end}")
    print("=" * 50)

    access_token, client_id, client_secret = authenticate()

    out_dir = Path(__file__).parent / "data" / "whoop"
    ensure_dir(out_dir)

    # WHOOP API expects ISO 8601 with time
    start_iso = f"{args.start}T00:00:00.000Z"
    end_iso = f"{args.end}T23:59:59.999Z"

    counts = {}

    print("\nFetching recovery data...")
    recovery = fetch_all_pages("/v1/recovery", access_token, start_iso, end_iso)
    counts["recovery"] = download_recovery(recovery, out_dir / "recovery.csv")

    print("Fetching sleep data...")
    sleep = fetch_all_pages("/v1/activity/sleep", access_token, start_iso, end_iso)
    counts["sleep"] = download_sleep(sleep, out_dir / "sleep.csv")

    print("Fetching workout data...")
    workouts = fetch_all_pages("/v1/activity/workout", access_token, start_iso, end_iso)
    counts["workouts"] = download_workouts(workouts, out_dir / "workouts.csv")

    print("Fetching cycle (strain) data...")
    cycles = fetch_all_pages("/v1/cycle", access_token, start_iso, end_iso)
    counts["cycles"] = download_cycles(cycles, out_dir / "cycles.csv")

    # Summary
    print("\n" + "=" * 50)
    print("Download complete — row counts:")
    print("=" * 50)
    for name, count in counts.items():
        print(f"  {name:<20} {count:>6} rows")
    print("=" * 50)


if __name__ == "__main__":
    main()
