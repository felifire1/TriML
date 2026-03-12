"""
TriML Athlete Year Tracker — Streamlit app.

Explore any of the 1,000 synthetic triathlete profiles:
  Panel 1 – Recovery Signals (HRV + Resting HR + 14-day HRV baseline)
  Panel 2 – Sleep (deep / REM / light stacked + sleep quality line)
  Panel 3 – Stress & Body Battery (AM vs PM shaded + stress line)
  Panel 4 – Training Load (TSS bars color-coded by sport + 7d/28d rolling + injury markers)
  Panel 5 – Activity Log (raw session table for selected date range)

Run:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.loader import (
    aggregate_activities,
    build_merged,
    ensure_data,
    find_demo_athlete,
    load_activities,
    load_athletes,
    load_daily,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPORT_COLORS = {
    "bike": "#2196F3",
    "run": "#FF9800",
    "swim": "#4CAF50",
    "strength": "#9E9E9E",
    "rest": "#E0E0E0",
}

INJURY_COLOR = "rgba(220,50,50,0.45)"


# ---------------------------------------------------------------------------
# Data loading (cached so it only runs once per session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _get_data_dir() -> str:
    """
    Ensure CSVs exist (downloads from Zenodo if missing).
    Returns data_dir as a string so it is safely hashable by st.cache_data.
    Cached for the server lifetime so downloads only happen once per deploy.
    """
    # Progress is printed to server logs on Cloud (no st widgets inside cache_resource)
    def _progress(filename, done, total):
        pct = done / total * 100
        print(f"  {filename}: {pct:.0f}%", flush=True)

    data_dir = ensure_data(progress_callback=_progress)
    return str(data_dir)


@st.cache_data(show_spinner=False)
def load_all_data(data_dir: str):
    """Load and parse all CSVs. data_dir must be passed so cache key is correct."""
    from pathlib import Path
    p = Path(data_dir)
    ath = load_athletes(p / "athletes.csv")
    daily = load_daily(p / "daily_data.csv")
    act = load_activities(p / "activity_data.csv")
    act_agg = aggregate_activities(act)
    merged = build_merged(daily, act_agg, ath)
    demo_id = find_demo_athlete(merged, act)
    return merged, act, ath, demo_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _injury_vlines(fig, injury_dates, secondary_y=False):
    for d in injury_dates:
        fig.add_vline(x=d, line_dash="dot", line_color=INJURY_COLOR, line_width=1)


def _athlete_profile(ath_row):
    st.divider()
    st.subheader("Athlete Profile")
    col_a, col_b = st.columns(2)
    col_a.metric("Age", int(ath_row["age"]))
    col_b.metric("Gender", ath_row["gender"].title())
    col_c, col_d = st.columns(2)
    col_c.metric("VO2max", f"{ath_row['vo2max']:.1f}")
    col_d.metric("FTP (W)", f"{ath_row['ftp']:.0f}")
    col_e, col_f = st.columns(2)
    col_e.metric("HRV Baseline", f"{ath_row['hrv_baseline']:.1f}")
    col_f.metric("Baseline RHR", f"{ath_row['resting_hr']:.0f} bpm")
    col_g, col_h = st.columns(2)
    col_g.metric("Training Exp.", f"{ath_row['training_experience']:.1f} yrs")
    col_h.metric("Wkly Hours", f"{ath_row['weekly_training_hours']:.1f}")
    st.caption(f"**Lifestyle:** {ath_row['lifestyle']}")


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="TriML — Athlete Tracker",
        layout="wide",
        page_icon="🏊",
        initial_sidebar_state="expanded",
    )

    st.title("TriML — Athlete Year Tracker")
    st.caption(
        "1,000 synthetic triathletes · 366 days · health signals, training load, and activities."
    )

    # Ensure CSVs exist — downloads from Zenodo on first run (Cloud or local)
    with st.spinner("Checking data files…"):
        data_dir = _get_data_dir()

    # Load + parse data
    with st.spinner("Loading dataset — this takes ~30 s on first run…"):
        merged, act, ath, demo_id = load_all_data(data_dir)

    athlete_ids = sorted(merged["athlete_id"].unique())
    default_idx = athlete_ids.index(demo_id) if demo_id in athlete_ids else 0

    # ------------------------------------------------------------------ Sidebar
    with st.sidebar:
        st.header("Controls")

        selected_id = st.selectbox(
            "Select athlete",
            athlete_ids,
            index=default_idx,
            help="Default is the most-injured athlete with all 4 sport types.",
        )

        all_dates = merged["date"].dt.date
        min_d, max_d = all_dates.min(), all_dates.max()

        date_range = st.slider(
            "Date range",
            min_value=min_d,
            max_value=max_d,
            value=(min_d, max_d),
            format="MMM DD",
        )

        show_injuries = st.toggle("Show injury markers", value=True)
        show_rolling = st.toggle("Show rolling averages", value=True)

        ath_row = ath[ath["athlete_id"] == selected_id].iloc[0]
        _athlete_profile(ath_row)

    # ------------------------------------------------------------- Filter data
    df = merged[
        (merged["athlete_id"] == selected_id)
        & (merged["date"].dt.date >= date_range[0])
        & (merged["date"].dt.date <= date_range[1])
    ].copy().reset_index(drop=True)

    act_df = act[
        (act["athlete_id"] == selected_id)
        & (act["date"].dt.date >= date_range[0])
        & (act["date"].dt.date <= date_range[1])
    ].copy()

    injury_dates = df.loc[df["injury"] == 1, "date"].tolist()

    # ---------------------------------------- Summary stats strip (top)
    n_days = len(df)
    n_injury = int(df["injury"].sum())
    total_tss = df["tss"].sum()
    avg_hrv = df["hrv"].mean()
    avg_sleep = df["sleep_hours"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Days shown", n_days)
    c2.metric("Injury days", f"{n_injury} ({100*n_injury/n_days:.1f}%)")
    c3.metric("Total TSS", f"{total_tss:,.0f}")
    c4.metric("Avg HRV", f"{avg_hrv:.1f}")
    c5.metric("Avg Sleep", f"{avg_sleep:.1f} h")

    st.divider()

    # ----------------------------------- PANEL 1: Recovery Signals (HRV + RHR)
    st.subheader("① Recovery Signals")

    fig1 = make_subplots(specs=[[{"secondary_y": True}]])

    if show_rolling:
        hrv_14d = df["hrv"].rolling(14, min_periods=3).mean()
        fig1.add_trace(
            go.Scatter(
                x=df["date"],
                y=hrv_14d,
                name="HRV 14-day baseline",
                line=dict(dash="dash", color="#B39DDB", width=1),
                opacity=0.7,
            ),
            secondary_y=False,
        )

    fig1.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["hrv"],
            name="HRV",
            line=dict(color="#7E57C2", width=2),
            hovertemplate="<b>HRV</b>: %{y:.1f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig1.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["resting_hr"],  # daily measured RHR from daily_data
            name="Resting HR",
            line=dict(color="#EF5350", width=2),
            hovertemplate="<b>RHR</b>: %{y:.1f} bpm<extra></extra>",
        ),
        secondary_y=True,
    )

    if show_injuries:
        _injury_vlines(fig1, injury_dates)

    fig1.update_yaxes(title_text="HRV", secondary_y=False)
    fig1.update_yaxes(title_text="Resting HR (bpm)", secondary_y=True)
    fig1.update_layout(
        height=280,
        margin=dict(t=10, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.15),
    )
    st.plotly_chart(fig1, use_container_width=True)

    # --------------------------------------------------- PANEL 2: Sleep
    st.subheader("② Sleep Architecture")

    fig2 = make_subplots(specs=[[{"secondary_y": True}]])

    for col, label, color in [
        ("deep_sleep", "Deep", "#1565C0"),
        ("rem_sleep", "REM", "#42A5F5"),
        ("light_sleep", "Light", "#90CAF9"),
    ]:
        fig2.add_trace(
            go.Bar(
                x=df["date"],
                y=df[col],
                name=label,
                marker_color=color,
                opacity=0.85,
                hovertemplate=f"<b>{label}</b>: %{{y:.2f}} h<extra></extra>",
            ),
            secondary_y=False,
        )

    fig2.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["sleep_quality"],
            name="Sleep quality",
            line=dict(color="#FFA726", width=2),
            hovertemplate="<b>Quality</b>: %{y:.2f}<extra></extra>",
        ),
        secondary_y=True,
    )

    if show_injuries:
        _injury_vlines(fig2, injury_dates)

    fig2.update_layout(
        barmode="stack",
        height=280,
        margin=dict(t=10, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.15),
    )
    fig2.update_yaxes(title_text="Hours", secondary_y=False)
    fig2.update_yaxes(title_text="Quality (0–1)", secondary_y=True, range=[0, 1.2])
    st.plotly_chart(fig2, use_container_width=True)

    # ---------------------------------------- PANEL 3: Stress & Body Battery
    st.subheader("③ Stress & Body Battery")

    fig3 = make_subplots(specs=[[{"secondary_y": True}]])

    fig3.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["body_battery_morning"],
            name="Battery AM",
            fill=None,
            line=dict(color="#66BB6A", width=1.5),
            hovertemplate="<b>Battery AM</b>: %{y:.0f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig3.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["body_battery_evening"],
            name="Battery PM",
            fill="tonexty",
            line=dict(color="#2E7D32", width=1.5),
            fillcolor="rgba(102,187,106,0.18)",
            hovertemplate="<b>Battery PM</b>: %{y:.0f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig3.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["stress"],
            name="Stress",
            line=dict(color="#EF5350", width=2),
            hovertemplate="<b>Stress</b>: %{y:.1f}<extra></extra>",
        ),
        secondary_y=True,
    )

    if show_injuries:
        _injury_vlines(fig3, injury_dates)

    fig3.update_yaxes(title_text="Body Battery (0–100)", secondary_y=False)
    fig3.update_yaxes(title_text="Stress score", secondary_y=True)
    fig3.update_layout(
        height=280,
        margin=dict(t=10, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.15),
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ------------------------------------------ PANEL 4: Training Load (TSS)
    st.subheader("④ Training Load")

    fig4 = go.Figure()

    # TSS bars grouped by dominant sport (one trace per sport for legend)
    for sport, color in SPORT_COLORS.items():
        mask = df["dominant_sport"] == sport
        if not mask.any():
            continue
        fig4.add_trace(
            go.Bar(
                x=df.loc[mask, "date"],
                y=df.loc[mask, "tss"],
                name=sport.title(),
                marker_color=color,
                hovertemplate=(
                    f"<b>{sport.title()}</b><br>"
                    "TSS: %{y:.1f}<br>"
                    "Date: %{x}<extra></extra>"
                ),
            )
        )

    if show_rolling:
        rolling7 = df["tss"].rolling(7, min_periods=1).mean()
        rolling28 = df["tss"].rolling(28, min_periods=7).mean()
        fig4.add_trace(
            go.Scatter(
                x=df["date"],
                y=rolling7,
                name="7-day avg (Acute)",
                line=dict(color="#FF6F00", width=2),
                hovertemplate="<b>Acute 7d</b>: %{y:.1f}<extra></extra>",
            )
        )
        fig4.add_trace(
            go.Scatter(
                x=df["date"],
                y=rolling28,
                name="28-day avg (Chronic)",
                line=dict(color="#B71C1C", width=2, dash="dash"),
                hovertemplate="<b>Chronic 28d</b>: %{y:.1f}<extra></extra>",
            )
        )

    if show_injuries:
        inj_rows = df[df["injury"] == 1]
        if not inj_rows.empty:
            fig4.add_trace(
                go.Scatter(
                    x=inj_rows["date"],
                    y=inj_rows["tss"] + 8,
                    mode="markers",
                    marker=dict(symbol="triangle-down", size=10, color="red"),
                    name="Injury day",
                    hovertemplate="<b>INJURY</b><br>Date: %{x}<extra></extra>",
                )
            )

    fig4.update_layout(
        barmode="stack",
        height=310,
        margin=dict(t=10, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.15),
    )
    fig4.update_yaxes(title_text="TSS")
    st.plotly_chart(fig4, use_container_width=True)

    # ----------------------------------------------- PANEL 5: Activity Table
    st.subheader("⑤ Activity Log")

    display_cols = [
        "date", "sport", "workout_type", "duration_minutes",
        "tss", "avg_hr", "intensity_factor", "distance_km",
    ]
    act_display = act_df[display_cols].copy()
    act_display["date"] = act_display["date"].dt.strftime("%Y-%m-%d")
    act_display = act_display.sort_values(["date", "sport"]).reset_index(drop=True)
    act_display.columns = [
        "Date", "Sport", "Workout Type", "Duration (min)",
        "TSS", "Avg HR", "Intensity", "Dist (km)",
    ]

    for col in ["TSS", "Avg HR", "Intensity", "Dist (km)"]:
        act_display[col] = act_display[col].apply(
            lambda x: f"{x:.1f}" if pd.notna(x) and x != 0 else "—"
        )

    def _sport_row_color(row):
        sport = row["Sport"]
        color_hex = SPORT_COLORS.get(sport, "#FFFFFF")
        # Convert hex to rgba with low alpha for row background
        r = int(color_hex[1:3], 16)
        g = int(color_hex[3:5], 16)
        b = int(color_hex[5:7], 16)
        bg = f"rgba({r},{g},{b},0.12)"
        return [f"background-color: {bg}"] * len(row)

    st.dataframe(
        act_display.style.apply(_sport_row_color, axis=1),
        use_container_width=True,
        height=320,
    )

    st.caption(
        "🔴 Red dashed lines = injury days  ·  "
        "🔵 Blue bars = bike  ·  🟠 Orange = run  ·  "
        "🟢 Green = swim  ·  ⚫ Gray = strength"
    )


if __name__ == "__main__":
    main()
