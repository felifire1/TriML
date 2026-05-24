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


# Load ML daily features via garmin_mapper

def _load_ml_daily():
    """Run garmin_mapper to get Grit Score + rolling features per day."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    try:
        from FutureSteps.garmin_mapper import build_personal_dataset
        df = build_personal_dataset()
        df["date_str"] = df["date"].astype(str).str[:10]
        return df.set_index("date_str").to_dict("index")
    except Exception as exc:
        print(f"  Warning: could not load garmin_mapper ({exc})")
        return {}


def _load_injury_model():
    """Load trained injury + load models from results/trained_models.pkl."""
    model_path = DATA_DIR.parent / "results" / "trained_models.pkl"
    if not model_path.exists():
        return None
    try:
        import pickle
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        print(f"  Warning: could not load trained models ({exc})")
        return None


def _compute_rolling_weekly_hours():
    """Compute a 4-week rolling avg of weekly training hours from Garmin activities."""
    import csv
    path = DATA_DIR / "garmin" / "activities.csv"
    if not path.exists():
        return {}
    try:
        from datetime import timedelta
        import collections
        # sum duration_seconds per day
        daily_seconds = collections.defaultdict(float)
        with open(path) as f:
            for row in csv.DictReader(f):
                d = (row.get("date") or "")[:10]
                try:
                    daily_seconds[d] += float(row.get("duration_seconds") or 0)
                except ValueError:
                    pass

        # for each date, compute avg weekly hours over trailing 4 weeks
        rolling = {}
        dates = sorted(daily_seconds.keys())
        from datetime import date as date_cls
        for d_str in dates:
            d = date_cls.fromisoformat(d_str)
            window_start = (d - timedelta(weeks=4)).isoformat()
            total_sec = sum(
                v for k, v in daily_seconds.items()
                if window_start <= k <= d_str
            )
            rolling[d_str] = (total_sec / 3600) / 4  # avg hours/week over 4 weeks
        return rolling
    except Exception:
        return {}


# precompute once at module load
_WEEKLY_HOURS_BY_DATE = None


def _get_weekly_hours(act_date):
    global _WEEKLY_HOURS_BY_DATE
    if _WEEKLY_HOURS_BY_DATE is None:
        _WEEKLY_HOURS_BY_DATE = _compute_rolling_weekly_hours()
    # fall back to .env value or 10h if no Garmin data
    fallback = float(os.environ.get("ATHLETE_WEEKLY_HOURS") or 10)
    return _WEEKLY_HOURS_BY_DATE.get(act_date, fallback)


def _athlete_profile_features(act_date=None):
    """Static profile from .env + dynamic weekly hours from Garmin activities."""
    gender_enc = 0 if os.environ.get("ATHLETE_GENDER", "male").lower() == "male" else 1
    lifestyle_map = {"recreational": 0, "amateur": 1, "professional": 2}
    lifestyle_enc = lifestyle_map.get(os.environ.get("ATHLETE_LIFESTYLE", "amateur").lower(), 1)
    return {
        "age": float(os.environ.get("ATHLETE_AGE") or 28),
        "vo2max": float(os.environ.get("ATHLETE_VO2MAX") or 52),
        "ftp": float(os.environ.get("ATHLETE_FTP") or 220),
        "training_experience": float(os.environ.get("ATHLETE_TRAINING_EXPERIENCE_YEARS") or 3),
        "weekly_training_hours": _get_weekly_hours(act_date),
        "gender_enc": gender_enc,
        "lifestyle_enc": lifestyle_enc,
    }


def _predict_injury(row, models, act_date=None, cumulative_fraction=1.0):
    """
    Run injury + load prediction for a single day's feature row.

    cumulative_fraction (0-1) scales the load-sensitive features (TSS,
    duration, ACWR) to reflect how far into the day's training we are.
    ACWR moves less than TSS because it's a 7-day rolling mean — today's
    partial load only nudges the numerator slightly.
    """
    import numpy as np

    feat_names = models["feature_names"]
    profile = _athlete_profile_features(act_date)

    # features that build up through the day
    load_features = {"tss", "duration_minutes"}

    # ACWR is a 7-day rolling mean, so today's partial load only moves it
    # by 1/7 of the day's contribution — much smaller effect
    acwr_end = row.get("acwr")
    chronic = None
    if acwr_end is not None:
        try:
            # approximate chronic from end-of-day acwr and tss
            tss_end = float(row.get("tss") or 0)
            acwr_f = float(acwr_end)
            # acute_end ≈ acwr * chronic, and acute includes today fully
            # partial acute = acute_end - tss_end/7 + (tss_end * fraction)/7
            # partial_acwr = partial_acute / chronic = acwr_end + (fraction-1) * tss_end / (7 * chronic)
            # we can express this without knowing chronic:
            # partial_acwr ≈ acwr_end * (1 - (1 - fraction) * tss_end / (7 * acwr_end * chronic))
            # simplify: just linearly blend acwr toward 1.0 (no load) as fraction decreases
            # acwr_partial = 1.0 + (acwr_end - 1.0) * fraction  (1.0 = no-load baseline)
            acwr_partial = 1.0 + (acwr_f - 1.0) * cumulative_fraction
        except (TypeError, ValueError):
            acwr_partial = acwr_end
    else:
        acwr_partial = acwr_end

    feat_vec = []
    for f in feat_names:
        val = row.get(f, profile.get(f, 0))
        try:
            v = float(val) if val is not None else 0.0
        except (TypeError, ValueError):
            v = 0.0

        if f in load_features:
            v = v * cumulative_fraction
        elif f == "acwr" and acwr_partial is not None:
            try:
                v = float(acwr_partial)
            except (TypeError, ValueError):
                pass

        feat_vec.append(v)

    X = models["scaler"].transform([feat_vec])
    injury_prob = models["injury_clf"].predict_proba(X)[0][1]
    load_pred = models["load_clf"].predict(X)[0]
    load_label = models["load_classes"][int(load_pred)]
    return injury_prob, load_label


# Format the comment

def _fmt(val, decimals=0, suffix=""):
    try:
        v = float(val)
        if decimals == 0:
            return f"{int(round(v))}{suffix}"
        return f"{v:.{decimals}f}{suffix}"
    except (TypeError, ValueError):
        return "—"


def _activity_effort(act):
    """Effort proxy for a Strava activity (used to weight cumulative Grit Score)."""
    se = act.get("suffer_score")
    if se:
        return float(se)
    # fallback: duration * normalized HR
    dur = act.get("moving_time") or act.get("elapsed_time") or 0
    avg_hr = act.get("average_heartrate") or 140
    return (dur / 60) * (avg_hr / 150)


def compute_effort_fractions(activities):
    """
    For each date with multiple activities, compute the cumulative effort
    fraction at each activity (sorted by start time).
    Returns {activity_id: cumulative_fraction (0-1)}
    """
    from collections import defaultdict
    by_date = defaultdict(list)
    for act in activities:
        ts = act.get("start_date_local", "")
        d = ts[:10] if ts else None
        if d:
            by_date[d].append(act)

    fractions = {}
    for d, acts in by_date.items():
        acts_sorted = sorted(acts, key=lambda a: a.get("start_date_local", ""))
        efforts = [_activity_effort(a) for a in acts_sorted]
        total = sum(efforts) or 1
        cumulative = 0
        for act, effort in zip(acts_sorted, efforts):
            cumulative += effort
            fractions[act["id"]] = cumulative / total
    return fractions


def _partial_grit(grit_end, acwr, cumulative_fraction):
    """
    Scale the load-sensitive component of Grit Score by how much of the
    day's training load has been done.

    The ACWR sub-score carries 15% of the total Grit Score weight.
    At the start of the day (no workouts done) that component is 0.
    By end of day it reaches its full value.
    """
    if grit_end is None:
        return None
    try:
        acwr_val = float(acwr) if acwr is not None else 1.0
        acwr_sub = min(abs(acwr_val - 1.0), 1.0)
        load_contribution_eod = 100 * 0.15 * acwr_sub
        load_contribution_now = load_contribution_eod * cumulative_fraction
        grit_now = float(grit_end) - load_contribution_eod + load_contribution_now
        return max(0.0, min(100.0, grit_now))
    except (TypeError, ValueError):
        return grit_end


def _activity_line(act):
    """Build a workout-specific stats line from the Strava activity object."""
    sport = (act.get("sport_type") or act.get("type") or "").lower()
    parts = []

    dist_m = act.get("distance")
    duration_s = act.get("moving_time") or act.get("elapsed_time")
    avg_hr = act.get("average_heartrate")
    max_hr = act.get("max_heartrate")
    avg_watts = act.get("average_watts")
    w_avg_watts = act.get("weighted_average_watts")
    elevation = act.get("total_elevation_gain")

    if dist_m:
        if "swim" in sport:
            parts.append(f"{dist_m:.0f}m")
        elif dist_m >= 1000:
            parts.append(f"{dist_m/1000:.1f}km")

    if duration_s and dist_m and dist_m > 0:
        if any(x in sport for x in ["run", "walk", "hike"]):
            pace_sec_km = duration_s / (dist_m / 1000)
            mins = int(pace_sec_km // 60)
            secs = int(pace_sec_km % 60)
            parts.append(f"{mins}:{secs:02d}/km")
        elif any(x in sport for x in ["ride", "cycl", "bike", "virtual"]):
            if w_avg_watts:
                parts.append(f"{w_avg_watts:.0f}w NP")
            elif avg_watts:
                parts.append(f"{avg_watts:.0f}w avg")

    if avg_hr:
        parts.append(f"{avg_hr:.0f}bpm avg")
    if max_hr:
        parts.append(f"{max_hr:.0f} max")
    if elevation and elevation > 10:
        parts.append(f"+{elevation:.0f}m")

    return "  |  ".join(parts) if parts else None


def build_comment(act_date, ml_daily, models, act=None, cumulative_fraction=1.0):
    row = ml_daily.get(act_date)
    if not row:
        return None

    grit_eod = row.get("grit_score")
    acwr = row.get("acwr")
    hrv_z = row.get("hrv_zscore")
    load_class = row.get("load_class", "Balanced")

    # Grit Score adjusted for how far into the day this activity is
    grit = _partial_grit(grit_eod, acwr, cumulative_fraction)

    lines = ["GritML"]

    if act:
        act_line = _activity_line(act)
        if act_line:
            lines.append(act_line)

    # Injury prediction
    if models:
        try:
            injury_prob, load_pred = _predict_injury(row, models, act_date, cumulative_fraction)
            risk_label = "High" if injury_prob > 0.5 else "Moderate" if injury_prob > 0.25 else "Low"
            lines.append(f"Injury Risk: {risk_label} ({injury_prob*100:.0f}%)  |  Load: {load_pred}")
        except Exception:
            pass

    # Grit Score
    if grit is not None:
        lines.append(f"Grit Score: {_fmt(grit)}/100  |  {load_class}")

    # ACWR + HRV
    details = []
    if acwr is not None:
        details.append(f"ACWR: {_fmt(acwr, 2)}")
    if hrv_z is not None:
        sign = "+" if float(hrv_z) >= 0 else ""
        details.append(f"HRV: {sign}{_fmt(hrv_z, 1)}sigma")
    if details:
        lines.append("  ".join(details))

    if len(lines) == 1:
        return None

    return "\n".join(lines)


# Main

def parse_args():
    parser = argparse.ArgumentParser(description="Post TriML ML stats as comments on Strava activities.")
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

    print("\nLoading ML daily features from Garmin data...")
    ml_daily = _load_ml_daily()
    if not ml_daily:
        print("  No Garmin data found. Run garmin_download.py first.")
        sys.exit(1)
    print(f"  {len(ml_daily)} days loaded.")

    models = _load_injury_model()
    if models:
        print("  Trained injury model loaded — will predict injury risk.")
    else:
        print("  No trained models found — run ml_pipeline.py first for injury prediction.")
        print("  (Grit Score + ACWR will still be posted.)")

    activities = fetch_activities(access_token, args.start, args.end)

    # pre-compute cumulative effort fractions so same-day activities
    # get progressively higher Grit Scores
    effort_fractions = compute_effort_fractions(activities)

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

        cumulative_fraction = effort_fractions.get(act_id, 1.0)
        comment = build_comment(act_date, ml_daily, models, act=act,
                                cumulative_fraction=cumulative_fraction)

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
