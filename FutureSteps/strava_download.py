#!/usr/bin/env python3
# FutureSteps: strava_download.py
"""
Strava annotator for TriML project.

Reads your downloaded WHOOP + Garmin CSVs and posts a stats comment
on each Strava activity for the matching date.

Run Garmin + WHOOP downloaders first, then:
    python3 strava_download.py
    python3 strava_download.py --start 2025-01-01 --end 2026-03-31
"""

import argparse
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
SCOPES = "activity:read_all,activity:write"

TOKEN_DIR = Path.home() / ".strava_tokens"
TOKEN_FILE = TOKEN_DIR / "tokens.json"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# Token helpers

def _ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def _save_tokens(tokens):
    _ensure_dir(TOKEN_DIR)
    with open(TOKEN_FILE, "w") as f:
        json.dump(tokens, f)


def _load_tokens():
    if TOKEN_FILE.exists():
        with open(TOKEN_FILE) as f:
            return json.load(f)
    return None


# OAuth2 flow

def _get_auth_code(client_id):
    params = {
        "client_id": client_id,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "approval_prompt": "force",
        "scope": SCOPES,
    }
    auth_url = f"{AUTH_URL}?{urllib.parse.urlencode(params)}"

    auth_code = None

    class CallbackHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            nonlocal auth_code
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            auth_code = qs.get("code", [None])[0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<h1>Authorized! You can close this tab.</h1>")

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

    print("Authorized!")
    return auth_code


def _exchange_code(client_id, client_secret, code):
    resp = requests.post(TOKEN_URL, data={
        "grant_type": "authorization_code",
        "code": code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": REDIRECT_URI,
    })
    resp.raise_for_status()
    tokens = resp.json()
    _save_tokens(tokens)
    return tokens


def _refresh_token(client_id, client_secret, refresh_token):
    resp = requests.post(TOKEN_URL, data={
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
        "client_secret": client_secret,
    })
    resp.raise_for_status()
    tokens = resp.json()
    _save_tokens(tokens)
    return tokens


def authenticate():
    client_id = os.environ.get("STRAVA_CLIENT_ID", "").strip()
    client_secret = os.environ.get("STRAVA_CLIENT_SECRET", "").strip()

    if not client_id or not client_secret:
        print("ERROR: STRAVA_CLIENT_ID and STRAVA_CLIENT_SECRET must be set in .env")
        sys.exit(1)

    tokens = _load_tokens()

    if tokens and tokens.get("expires_at", 0) > time.time() + 60:
        print("Using saved Strava token.")
        return tokens["access_token"], client_id, client_secret

    if tokens and tokens.get("refresh_token"):
        try:
            print("Refreshing Strava token...")
            tokens = _refresh_token(client_id, client_secret, tokens["refresh_token"])
            print("Token refreshed!")
            return tokens["access_token"], client_id, client_secret
        except Exception as exc:
            print(f"Refresh failed ({exc}), re-authorizing...")

    code = _get_auth_code(client_id)
    tokens = _exchange_code(client_id, client_secret, code)
    return tokens["access_token"], client_id, client_secret


# Strava API helpers

def _api_get(endpoint, access_token, params=None):
    url = f"{API_BASE}{endpoint}"
    headers = {"Authorization": f"Bearer {access_token}"}
    for attempt in range(5):
        resp = requests.get(url, headers=headers, params=params)
        if resp.status_code == 429:
            wait = 60 * (attempt + 1)
            print(f"\n  Rate limited. Waiting {wait}s...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()
    raise Exception("Max retries exceeded")


def _api_post(endpoint, access_token, data):
    url = f"{API_BASE}{endpoint}"
    headers = {"Authorization": f"Bearer {access_token}"}
    resp = requests.post(url, headers=headers, json=data)
    resp.raise_for_status()
    return resp.json()


def fetch_activities(access_token, start_date, end_date):
    after = int(datetime.fromisoformat(f"{start_date}T00:00:00").replace(tzinfo=timezone.utc).timestamp())
    before = int(datetime.fromisoformat(f"{end_date}T23:59:59").replace(tzinfo=timezone.utc).timestamp())

    all_activities = []
    page = 1
    print(f"Fetching Strava activities ({start_date} -> {end_date})...")
    while True:
        batch = _api_get("/athlete/activities", access_token, params={
            "after": after, "before": before, "per_page": 100, "page": page,
        })
        if not batch:
            break
        all_activities.extend(batch)
        print(f"\r  {len(all_activities)} activities...", end="", flush=True)
        if len(batch) < 100:
            break
        page += 1
        time.sleep(0.3)
    print()
    print(f"  Found {len(all_activities)} activities.")
    return all_activities


# Load WHOOP + Garmin data into date-keyed dicts

def _load_whoop_recovery():
    path = DATA_DIR / "whoop" / "recovery.csv"
    if not path.exists():
        return {}
    import csv
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            d = row.get("date", "")[:10]
            if d:
                data[d] = row
    return data


def _load_whoop_sleep():
    path = DATA_DIR / "whoop" / "sleep.csv"
    if not path.exists():
        return {}
    import csv
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            d = row.get("date", "")[:10]
            if d:
                data[d] = row
    return data


def _load_garmin_daily():
    path = DATA_DIR / "garmin" / "daily_summary.csv"
    if not path.exists():
        return {}
    import csv
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            d = row.get("date", "")[:10]
            if d:
                data[d] = row
    return data


def _load_garmin_hrv():
    path = DATA_DIR / "garmin" / "hrv.csv"
    if not path.exists():
        return {}
    import csv
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            d = row.get("date", "")[:10]
            if d:
                data[d] = row
    return data


# Format the comment

def _fmt(val, decimals=0, suffix=""):
    try:
        v = float(val)
        if decimals == 0:
            return f"{int(round(v))}{suffix}"
        return f"{v:.{decimals}f}{suffix}"
    except (TypeError, ValueError):
        return "—"


def _ms_to_hours(val):
    try:
        return float(val) / 3_600_000
    except (TypeError, ValueError):
        return None


def build_comment(act_date, whoop_rec, whoop_sleep, garmin_daily, garmin_hrv):
    rec = whoop_rec.get(act_date, {})
    slp = whoop_sleep.get(act_date, {})
    grm = garmin_daily.get(act_date, {})
    hrv = garmin_hrv.get(act_date, {})

    lines = ["📊 TriML Stats"]

    # WHOOP recovery
    recovery = rec.get("recovery_score")
    hrv_whoop = rec.get("hrv_rmssd")
    sleep_ms = slp.get("total_sleep_ms")
    sleep_h = _ms_to_hours(sleep_ms)
    sleep_perf = slp.get("sleep_performance")

    if any(v not in (None, "", "None") for v in [recovery, hrv_whoop, sleep_h]):
        parts = []
        if recovery not in (None, "", "None"):
            parts.append(f"Recovery: {_fmt(recovery)}%")
        if hrv_whoop not in (None, "", "None"):
            parts.append(f"HRV: {_fmt(hrv_whoop)}ms")
        if sleep_h is not None:
            parts.append(f"Sleep: {_fmt(sleep_h, 1)}h")
        if sleep_perf not in (None, "", "None"):
            parts.append(f"Sleep perf: {_fmt(sleep_perf)}%")
        lines.append(" · ".join(parts))

    # Garmin daily
    rhr = grm.get("resting_hr")
    stress = grm.get("avg_stress")
    bb_high = grm.get("body_battery_high")
    bb_low = grm.get("body_battery_low")
    hrv_garmin = hrv.get("hrv_last_night") or hrv.get("hrv_weekly_avg")

    if any(v not in (None, "", "None") for v in [rhr, stress, bb_high, hrv_garmin]):
        parts = []
        if rhr not in (None, "", "None"):
            parts.append(f"RHR: {_fmt(rhr)}bpm")
        if hrv_garmin not in (None, "", "None"):
            parts.append(f"HRV: {_fmt(hrv_garmin)}ms")
        if bb_high not in (None, "", "None") and bb_low not in (None, "", "None"):
            parts.append(f"Battery: {_fmt(bb_low)}→{_fmt(bb_high)}")
        if stress not in (None, "", "None"):
            parts.append(f"Stress: {_fmt(stress)}")
        lines.append(" · ".join(parts))

    if len(lines) == 1:
        return None  # no data for this date

    return "\n".join(lines)


# Main

def parse_args():
    parser = argparse.ArgumentParser(description="Post TriML stats as comments on Strava activities.")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument("--dry-run", action="store_true", help="Print comments without posting")
    return parser.parse_args()


def main():
    args = parse_args()
    print("Strava TriML Annotator")
    print(f"Date range: {args.start} -> {args.end}")
    if args.dry_run:
        print("DRY RUN — nothing will be posted")
    print("=" * 50)

    access_token, _, _ = authenticate()

    print("\nLoading WHOOP + Garmin data...")
    whoop_rec = _load_whoop_recovery()
    whoop_sleep = _load_whoop_sleep()
    garmin_daily = _load_garmin_daily()
    garmin_hrv = _load_garmin_hrv()

    sources = []
    if whoop_rec:
        sources.append(f"WHOOP recovery ({len(whoop_rec)} days)")
    if garmin_daily:
        sources.append(f"Garmin daily ({len(garmin_daily)} days)")
    if not sources:
        print("  No WHOOP or Garmin data found. Run those downloaders first.")
        sys.exit(1)
    print(f"  Loaded: {', '.join(sources)}")

    activities = fetch_activities(access_token, args.start, args.end)

    posted = 0
    skipped = 0

    print(f"\nAnnotating {len(activities)} activities...")
    for act in activities:
        start_ts = act.get("start_date_local", "")
        act_date = start_ts[:10] if start_ts else None
        act_id = act.get("id")
        act_name = act.get("name", "")

        if not act_date or not act_id:
            continue

        comment = build_comment(act_date, whoop_rec, whoop_sleep, garmin_daily, garmin_hrv)

        if not comment:
            skipped += 1
            continue

        if args.dry_run:
            print(f"\n  [{act_date}] {act_name}")
            print("  " + comment.replace("\n", "\n  "))
        else:
            try:
                _api_post(f"/activities/{act_id}/comments", access_token, {"text": comment})
                print(f"  ✓ [{act_date}] {act_name}")
                posted += 1
                time.sleep(0.5)  # be nice to the API
            except Exception as exc:
                print(f"  ✗ [{act_date}] {act_name}: {exc}")

    print(f"\n{'=' * 50}")
    if args.dry_run:
        print(f"  Would post on {len(activities) - skipped} activities ({skipped} skipped — no data)")
    else:
        print(f"  Posted: {posted} · Skipped (no data): {skipped}")
    print("=" * 50)


if __name__ == "__main__":
    main()
