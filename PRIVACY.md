# Privacy Policy — TriML

**Last updated:** March 31, 2026

## What data we collect
TriML connects to fitness platforms (Garmin Connect, WHOOP, TrainingPeaks, Strava) via their official APIs. When you authorize access, we retrieve:
- Daily health metrics (HRV, resting heart rate, sleep, stress, body battery)
- Training activities (sport, duration, distance, heart rate, power, TSS)
- Planned workouts (from TrainingPeaks)

## How we use it
Your data is used solely to:
- Compute your personal Grit Score (injury risk metric)
- Generate training load analysis and recommendations
- Optionally write adjusted workouts back to TrainingPeaks or activity summaries to Strava (only with your explicit permission)

## Data storage
- All data is stored locally on your device or in your private account
- We do not sell, share, or transfer your data to third parties
- OAuth tokens are stored locally and never transmitted to external servers

## Data deletion
You can revoke access at any time through each platform's settings:
- WHOOP: app.whoop.com → Settings → Connected Apps
- Garmin: connect.garmin.com → Settings → Connected Apps
- TrainingPeaks: trainingpeaks.com → Settings → Authorized Applications
- Strava: strava.com → Settings → My API Application

Revoking access immediately stops all data retrieval. To delete previously downloaded data, remove the `data/` directory from your local TriML installation.

## Contact
Felipe Quiroz — github.com/felifire1
