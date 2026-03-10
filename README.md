# TriML — Predicting Athlete Training State from Wearable Data

**CS 6140: Machine Learning · Northeastern University**

## Overview

This project engineers a composite **Grit Score** from daily physiological signals (HRV, sleep, resting HR, training load) to classify triathletes into three readiness states: **Overreaching**, **Balanced**, and **Undertrained**.

**Dataset:** [Synthetic Triathlete Dataset (Rossi, 2025)](https://doi.org/10.5281/zenodo.15401061)
1,000 athletes · 366 days · 384,153 activity sessions

## Setup

```bash
pip install -r requirements.txt
```

The three CSV files (`athletes.csv`, `daily_data.csv`, `activity_data.csv`) should be placed in the project root. They are excluded from version control due to size.

## Athlete Year Tracker

Interactive Streamlit app to explore any athlete's health signals and training load:

```bash
streamlit run app/streamlit_app.py
```

Panels: Recovery Signals (HRV + RHR) · Sleep Architecture · Stress & Body Battery · Training Load (TSS by sport) · Activity Log

## Project Structure

```
TriML/
  src/
    loader.py       # Data loading and parsing
    features.py     # Grit Score and feature engineering (Phase 2)
    models.py       # PyTorch DNN models (Phase 2)
  app/
    streamlit_app.py  # Athlete tracker UI
  notebooks/
    01_eda.ipynb      # Exploratory data analysis
  requirements.txt
```

## Models (Phase 2)

| Task | Models |
|---|---|
| Classification (Overreaching / Balanced / Undertrained) | Logistic Regression, Random Forest, DNN (PyTorch) |
| Regression (continuous Grit Score 0–100) | Lasso + Polynomial Features, Random Forest, DNN (PyTorch) |

Evaluation: 5-fold stratified GroupKFold CV · AUC-ROC · F1-macro · RMSE · R²
