# TriML — Lovable Build Prompt

## Project Overview

Build a modern, interactive web dashboard called **TriML — Athlete Intelligence Platform** for exploring 1,000 synthetic triathlete profiles across 366 days of health signals, training load, and activity data. This is a data science course project (CS 6140) focused on injury prediction and athletic performance using machine learning.

---

## Data Source

Three CSV files, publicly available on Zenodo (record 15401061):

| File | Rows | Description |
|---|---|---|
| `athletes.csv` | 1,000 | Static athlete profiles |
| `daily_data.csv` | 366,000 | Daily wearable readings (1,000 athletes × 366 days, year 2024) |
| `activity_data.csv` | 384,153 | Individual training session records |

**Zenodo download URLs:**
- `https://zenodo.org/api/records/15401061/files/athletes.csv/content`
- `https://zenodo.org/api/records/15401061/files/daily_data.csv/content`
- `https://zenodo.org/api/records/15401061/files/activity_data.csv/content`

---

## Data Schema

### athletes.csv (~1,000 rows)
| Column | Type | Notes |
|---|---|---|
| athlete_id | string (UUID) | primary key |
| age | int | 18–65 |
| gender | string | male / female |
| sport_focus | string | triathlete / cyclist / runner / swimmer |
| experience_level | string | beginner / intermediate / advanced / elite |
| training_experience | float | years |
| weekly_training_hours | float | average hours/week |
| vo2max | float | ml/kg/min |
| ftp | float | functional threshold power (watts) |
| hrv_baseline | float | resting HRV baseline |
| resting_hr | float | baseline resting heart rate (bpm) — NOTE: renamed to `baseline_rhr` after merge to avoid collision with daily resting_hr |
| sleep_quality | float | 0–1 baseline — NOTE: renamed to `baseline_sleep_quality` after merge |
| lifestyle | string | professional / amateur / recreational |
| hrv_range | string | encoded as `(np.float64(82.9), np.float64(112.1))` — parse to hrv_min, hrv_max |
| hr_zones | string | dict with Z1–Z6 zone boundaries, contains np.float64() wrappers — parse with regex |

### daily_data.csv (~366,000 rows)
| Column | Type | Notes |
|---|---|---|
| athlete_id | string | foreign key |
| date | date | 2024-01-01 to 2024-12-31 |
| hrv | float | heart rate variability |
| resting_hr | float | actual daily resting HR (bpm) |
| sleep_hours | float | total sleep |
| deep_sleep | float | hours |
| rem_sleep | float | hours |
| light_sleep | float | hours |
| sleep_quality | float | 0–1 |
| stress | float | stress score |
| body_battery_morning | float | 0–100 |
| body_battery_evening | float | 0–100 |
| injury | int | 0 or 1 (injury flag) |
| tss | float | training stress score |

### activity_data.csv (~384,000 rows)
| Column | Type | Notes |
|---|---|---|
| athlete_id | string | foreign key |
| date | date | session date |
| sport | string | bike / run / swim / strength |
| workout_type | string | e.g. endurance, intervals, recovery |
| duration_minutes | float | |
| distance_km | float | |
| tss | float | training stress score for this session |
| avg_hr | float | average heart rate (bpm) |
| intensity_factor | float | 0–1 |
| work_kilojoules | float | |
| training_effect_aerobic | float | |
| training_effect_anaerobic | float | |
| hr_zones | string | `{'Z1': 0.12, 'Z2': 0.35, ...}` percent time per zone |
| power_zones | string | same format, null for non-cycling |

---

## What Was Already Built (Python/Streamlit — replicate and improve this)

### Data pipeline (`src/loader.py`)
1. **`load_athletes()`** — loads athletes.csv, parses `hrv_range` string with regex, parses `hr_zones` dict (removing `np.float64()` wrappers), expands into numeric columns
2. **`load_daily()`** — loads daily_data.csv, parses dates, sorts by (athlete_id, date)
3. **`load_activities()`** — loads activity_data.csv, parses `hr_zones` and `power_zones` dicts into 6 HR zone % columns and 7 power zone % columns
4. **`aggregate_activities()`** — collapses activity_data to one row per (athlete_id, date): sums TSS/duration/kilojoules, averages intensity/HR/zone distributions, counts sessions per sport, determines dominant_sport (highest-TSS sport that day), builds sports_of_day string
5. **`build_merged()`** — left-joins daily → activity aggregate → athlete profile into one 366,000-row analysis frame. Rest days get 0 for all activity columns. Renames athlete `resting_hr` → `baseline_rhr` and `sleep_quality` → `baseline_sleep_quality` to avoid collision.
6. **`find_demo_athlete()`** — finds the most-injured athlete who has all 4 sport types for use as default

### Dashboard (`app/streamlit_app.py`)
5-panel interactive athlete year tracker:

**Sidebar controls:**
- Athlete selector dropdown (UUID-based, 1,000 options)
- Date range slider (Jan–Dec 2024)
- Toggle: Show injury markers (red dashed vertical lines on all charts)
- Toggle: Show rolling averages
- Athlete profile card (age, gender, VO2max, FTP, HRV baseline, baseline RHR, training experience, weekly hours, lifestyle)

**Summary stats strip (top of page):**
- Days shown, Injury days (with %), Total TSS, Avg HRV, Avg Sleep

**Panel 1 — Recovery Signals:**
- Dual-axis Plotly line chart
- Left Y: HRV (purple line) + 14-day rolling baseline (dashed)
- Right Y: Resting HR (red line)
- Injury markers as vertical dotted lines

**Panel 2 — Sleep Architecture:**
- Stacked bar chart: Deep (dark blue) / REM (medium blue) / Light (light blue) sleep hours
- Secondary Y axis: Sleep quality score (0–1) as orange line

**Panel 3 — Stress & Body Battery:**
- Area fill between AM battery (light green) and PM battery (dark green)
- Secondary Y: Stress score (red line)

**Panel 4 — Training Load:**
- Bar chart colored by dominant sport: bike=blue, run=orange, swim=green, strength=gray
- Rolling overlays: 7-day acute (orange solid), 28-day chronic (dark red dashed)
- Injury markers as red triangle-down markers above bars

**Panel 5 — Activity Log:**
- Sortable table of individual sessions with sport-colored row backgrounds
- Columns: Date, Sport, Workout Type, Duration, TSS, Avg HR, Intensity, Distance

---

## What to Build Better

Replicate everything above, but improve it significantly with a modern React/TypeScript frontend. Key improvements:

### UX/UI
- **Replace UUID dropdown** with a smart search/filter: search by age range, gender, experience level, sport focus, injury rate slider — don't make users pick from a list of 1,000 UUIDs
- **Athlete card** with a visual profile: sport-type icon, colored experience badge, mini sparklines of HRV and TSS for the year
- **Dark mode** by default — this is health/sports data, dark looks better
- **Responsive layout** — works on tablet

### Dashboard Enhancements
- **Grit Score panel** (not yet built): compute and display a composite 0–100 score per day based on:
  - ACWR (acute:chronic workload ratio) = 7-day TSS / 28-day TSS
  - HRV deviation from personal 14-day baseline (as a z-score)
  - Sleep quality composite (hours × quality)
  - RHR 7-day trend (rising RHR = stress signal)
  - Normalize each 0–1, combine with equal weights → Grit Score
  - Color-code by zone: Overreaching (red, ACWR > 1.5), Balanced (green, 0.8–1.5), Undertrained (yellow, < 0.8)
- **Injury risk timeline** — show predicted injury risk as a heatmap calendar (like GitHub contribution graph) overlaid with actual injury days
- **Sport distribution donut chart** — breakdown of training time by sport for selected period
- **HR Zone distribution chart** — stacked bar showing % time in each HR zone across all sessions
- **Compare mode** — select 2 athletes side by side

### Data & Performance
- Pre-aggregate the data server-side; expose a REST API so the frontend doesn't need to load 186 MB CSVs into the browser
- Cache athlete summaries (yearly totals, averages) so the athlete selector is fast
- Load only the selected athlete's data on demand

### Tech Stack Suggestion for Lovable
- **Frontend:** React + TypeScript + Tailwind CSS
- **Charts:** Recharts or Nivo (both work well with React)
- **Backend:** FastAPI (Python) — reuse the existing loader.py data pipeline, add API endpoints
- **Data:** Load CSVs at startup, keep in memory, serve via API

---

## API Endpoints to Build (FastAPI backend)

```
GET /athletes                          → list of athletes with summary stats (for search/filter)
GET /athletes/{id}                     → full athlete profile
GET /athletes/{id}/daily?start=&end=   → daily_data rows for date range
GET /athletes/{id}/activities?start=&end= → activity rows for date range
GET /athletes/{id}/grit?start=&end=    → computed grit score per day
GET /athletes/{id}/summary             → yearly aggregates (total TSS, injury days, avg HRV, etc.)
```

---

## Important Notes
- All data is **synthetic** — no real athletes, safe to display publicly
- Injury days are already flagged as 0/1 in daily_data.csv — no inference needed for markers
- TSS is pre-computed per session in activity_data and per day in daily_data
- The `hrv_range` and `hr_zones` columns in athletes.csv are Python-repr strings with `np.float64()` wrappers — these need to be parsed with regex before use
- The `hr_zones` and `power_zones` columns in activity_data.csv are clean Python dict strings — parse with `ast.literal_eval`
