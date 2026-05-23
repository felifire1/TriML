#!/usr/bin/env python3
# FutureSteps: strava_download.py
"""
Strava API data downloader for TriML project.

Downloads activities via OAuth2 and saves to data/strava/.

Usage:
    python3 strava_download.py
    python3 strava_download.py --start 2025-01-01 --end 2026-03-31
"""

import argparse
import csv
import http.server
import json
import os
import sys
import time
import urllib.parse
import webbrowser
from datetime import date, datetime, timezone
from pathlib import Path

import requests


def _load_env():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip())

_load_env()

# Strava API config
AUTH_URL = "https://www.strava.com/oauth/authorize"
TOKEN_URL = "https://www.strava.com/oauth/token"
API_BASE = "https://www.strava.com/api/v3"
REDIRECT_URI = "http://localhost:8000/callback"
SCOPES = "activity:read_all,profile:read_all"

TOKEN_DIR = Path.home() / ".strava_tokens"
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
    params = {
        "client_id": client_id,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "approval_prompt": "auto",
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
            self.wfile.write(b"<h1>Success! You can close this tab.</h1>")

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("localhost", 8000), CallbackHandler)
    print("\nOpening browser for Strava login...")
    webbrowser.open(auth_url)
    print("Waiting for callback...")
    server.handle_request()
    server.server_close()

    if not auth_code:
        print("ERROR: No authorization code received.")
        sys.exit(1)

    print("Authorization code received!")
    return auth_code


def exchange_code(client_id, client_secret, code):
    resp = requests.post(TOKEN_URL, data={
        "grant_type": "authorization_code",
        "code": code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": REDIRECT_URI,
    })
    resp.raise_for_status()
    tokens = resp.json()
    save_tokens(tokens)
    print("Tokens saved!")
    return tokens


def refresh_access_token(client_id, client_secret, refresh_token):
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
    client_id = os.environ.get("STRAVA_CLIENT_ID", "").strip()
    client_secret = os.environ.get("STRAVA_CLIENT_SECRET", "").strip()

    if not client_id or not client_secret:
        print("ERROR: STRAVA_CLIENT_ID and STRAVA_CLIENT_SECRET must be set in .env")
        sys.exit(1)

    print("\nStrava Authentication")

    tokens = load_tokens()

    # Check if saved token is still valid
    if tokens and tokens.get("access_token"):
        expires_at = tokens.get("expires_at", 0)
        if expires_at > time.time() + 60:
            print("Using saved token.")
            return tokens["access_token"], client_id, client_secret

    # Try refresh
    if tokens and tokens.get("refresh_token"):
        try:
            print("Refreshing token...")
            tokens = refresh_access_token(client_id, client_secret, tokens["refresh_token"])
            print("Token refreshed!")
            return tokens["access_token"], client_id, client_secret
        except Exception as exc:
            print(f"Refresh failed ({exc}), starting fresh OAuth2 flow...")

    code = get_auth_code(client_id)
    tokens = exchange_code(client_id, client_secret, code)
    return tokens["access_token"], client_id, client_secret


# API helpers

def api_get(endpoint, access_token, params=None):
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


def fetch_all_activities(access_token, start_date, end_date):
    # Strava uses unix timestamps for filtering
    after = int(datetime.fromisoformat(f"{start_date}T00:00:00").replace(tzinfo=timezone.utc).timestamp())
    before = int(datetime.fromisoformat(f"{end_date}T23:59:59").replace(tzinfo=timezone.utc).timestamp())

    all_activities = []
    page = 1

    print(f"\nFetching activities ({start_date} -> {end_date})...")
    while True:
        batch = api_get("/athlete/activities", access_token, params={
            "after": after,
            "before": before,
            "per_page": 100,
            "page": page,
        })
        if not batch:
            break
        all_activities.extend(batch)
        print(f"\r  Page {page}: {len(all_activities)} activities so far...", end="", flush=True)
        if len(batch) < 100:
            break
        page += 1
        time.sleep(0.3)

    print()
    return all_activities


def download_activities(access_token, start_date, end_date, out_path):
    fieldnames = [
        "date", "name", "sport_type", "distance_meters", "duration_seconds",
        "moving_time_seconds", "elevation_gain", "avg_hr", "max_hr",
        "avg_power", "weighted_avg_power", "kilojoules",
        "avg_speed_ms", "max_speed_ms",
        "avg_cadence", "suffer_score", "calories",
        "training_effect_aerobic", "training_effect_anaerobic",
        "pr_count", "kudos_count", "strava_id",
    ]

    activities = fetch_all_activities(access_token, start_date, end_date)
    print(f"  Found {len(activities)} activities.")

    rows = []
    for a in activities:
        start_ts = a.get("start_date_local", "")
        act_date = start_ts[:10] if start_ts else None

        rows.append({
            "date": act_date,
            "name": a.get("name"),
            "sport_type": a.get("sport_type") or a.get("type"),
            "distance_meters": a.get("distance"),
            "duration_seconds": a.get("elapsed_time"),
            "moving_time_seconds": a.get("moving_time"),
            "elevation_gain": a.get("total_elevation_gain"),
            "avg_hr": a.get("average_heartrate"),
            "max_hr": a.get("max_heartrate"),
            "avg_power": a.get("average_watts"),
            "weighted_avg_power": a.get("weighted_average_watts"),
            "kilojoules": a.get("kilojoules"),
            "avg_speed_ms": a.get("average_speed"),
            "max_speed_ms": a.get("max_speed"),
            "avg_cadence": a.get("average_cadence"),
            "suffer_score": a.get("suffer_score"),
            "calories": a.get("calories"),
            "training_effect_aerobic": a.get("perceived_exertion"),
            "training_effect_anaerobic": None,
            "pr_count": a.get("pr_count"),
            "kudos_count": a.get("kudos_count"),
            "strava_id": a.get("id"),
        })

    _write_csv(out_path, fieldnames, rows)
    return len(rows)


def download_athlete_profile(access_token, out_path):
    fieldnames = [
        "athlete_id", "firstname", "lastname", "sex", "city", "state", "country",
        "weight_kg", "ftp", "follower_count", "friend_count",
    ]
    print("\nFetching athlete profile...")
    data = api_get("/athlete", access_token)

    rows = [{
        "athlete_id": data.get("id"),
        "firstname": data.get("firstname"),
        "lastname": data.get("lastname"),
        "sex": data.get("sex"),
        "city": data.get("city"),
        "state": data.get("state"),
        "country": data.get("country"),
        "weight_kg": data.get("weight"),
        "ftp": data.get("ftp"),
        "follower_count": data.get("follower_count"),
        "friend_count": data.get("friend_count"),
    }]

    _write_csv(out_path, fieldnames, rows)


# Main

def parse_args():
    parser = argparse.ArgumentParser(description="Download Strava data to CSV files.")
    parser.add_argument("--start", default="2025-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=date.today().isoformat(), help="End date (YYYY-MM-DD)")
    return parser.parse_args()


def main():
    args = parse_args()
    print("Strava Data Downloader")
    print(f"Date range: {args.start} -> {args.end}")
    print("=" * 50)

    access_token, client_id, client_secret = authenticate()

    out_dir = Path(__file__).parent.parent / "data" / "strava"
    ensure_dir(out_dir)

    download_athlete_profile(access_token, out_dir / "athlete_profile.csv")

    count = download_activities(access_token, args.start, args.end, out_dir / "activities.csv")

    print("\n" + "=" * 50)
    print(f"  activities         {count:>6} rows")
    print("=" * 50)
    print(f"\nFiles saved to: {out_dir}")


if __name__ == "__main__":
    main()
