# TriML -- Predicting Athlete Training State and Injury Risk from Wearable Sensor Data

**CS 6140: Machine Learning | Northeastern University | Spring 2026**

William Felipe Quiroz

## Overview

This project predicts an athlete's current training state (Overreaching / Balanced / Undertrained) and injury risk from daily wearable sensor data. We engineer a composite **Grit Score** from training load and recovery metrics, then apply 3 classifiers and 3 regressors with 5-fold GroupKFold cross-validation across 1,000 synthetic triathletes.

## Dataset

[Synthetic Triathlete Dataset for Injury Prediction Research (2024)](https://zenodo.org/records/15401061) -- Rossi, University of St. Gallen.

| File | Rows | Description |
|---|---|---|
| `athletes.csv` | 1,000 | Static athlete profiles (age, gender, VO2max, FTP, etc.) |
| `daily_data.csv` | 366,000 | Daily wearable readings (HRV, RHR, sleep, stress, body battery) |
| `activity_data.csv` | 384,153 | Individual training sessions (TSS, HR zones, power zones) |

Data is auto-downloaded from Zenodo on first run.

## Installation

```bash
git clone https://github.com/felifire1/TriML.git
cd TriML
pip install -r requirements.txt
```

### Requirements
- Python 3.10+
- PyTorch (for DNN models)
- scikit-learn, pandas, numpy, plotly, matplotlib, seaborn

## Reproducing Results

### 1. Run the ML pipeline (main results)
```bash
python3 ml_pipeline.py
```
Runs all 6 models with 5-fold GroupKFold CV. Takes ~30 min. Results saved to `results/ml_results.pkl`.

### 2. Run the hyperparameter sweep
```bash
python3 ml_pipeline.py --tune
```
Sweeps key hyperparameters for each model. Takes ~30 min. Results saved to `results/hp_sweep.pkl`.

### 3. Generate plots
```bash
python3 generate_plots.py
```
Generates 10 publication-quality plots to `results/plots/`.

### 4. Run the Streamlit dashboard (optional)
```bash
streamlit run app/streamlit_app.py
```
Interactive athlete year tracker at `http://localhost:8501`.

## Results Summary

### Injury Classification (binary)
| Model | ROC-AUC | F1-macro | Accuracy |
|---|---|---|---|
| Logistic Regression | 0.875 | 0.699 | 85.8% |
| Random Forest | 0.901 | 0.839 | 95.1% |
| **DNN (MLP)** | **0.948** | **0.871** | **95.5%** |

### Load Class Classification (3-class)
| Model | ROC-AUC | F1-macro | Accuracy |
|---|---|---|---|
| Logistic Regression | 0.988 | 0.914 | 91.3% |
| Random Forest | 0.987 | 0.914 | 91.3% |
| **DNN (MLP)** | **0.997** | **0.959** | **95.9%** |

### Grit Score Regression (0-100)
| Model | RMSE | R-squared |
|---|---|---|
| Lasso + Poly | 1.81 | 0.973 |
| Random Forest | 1.54 | 0.981 |
| **DNN (MLP)** | **1.19** | **0.988** |

## Project Structure

```
TriML/
  ml_pipeline.py          # Main ML pipeline (features + models + CV)
  generate_plots.py       # Plot generation from results
  requirements.txt        # Python dependencies
  src/
    loader.py             # Data loading + parsing (3 CSVs)
    features.py           # Feature engineering + Grit Score
    models.py             # Model definitions (sklearn + PyTorch)
  app/
    streamlit_app.py      # Interactive dashboard
  results/
    ml_results.pkl        # Saved CV metrics
    hp_sweep.pkl          # Hyperparameter sweep results
    pipeline_run.log      # Console output log
    plots/                # 10 PNG visualizations
  data/
    garmin/               # Personal Garmin Connect data (optional)
```

## References

1. Rossi, L. (2025). Synthetic Triathlete Dataset for Injury Prediction Research. Zenodo. https://doi.org/10.5281/zenodo.15401061
2. Gabbett, T.J. (2016). The training-injury prevention paradox. BJSM, 50(5), 273-280.
3. Hulin, B.T., et al. (2016). Spikes in acute workload are associated with increased injury risk. BJSM, 48(8), 708-712.
