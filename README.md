# TriML

**CS 6140: Machine Learning — Northeastern University — Spring 2026**

William Felipe Quiroz

## What is this

Predicting athlete training state and injury risk from wearable data. Uses a synthetic triathlete dataset (1000 athletes, ~366k daily records) to train classifiers and regressors. I engineered a composite "Grit Score" from recovery/training metrics, then used it alongside ACWR and other features.

**Models:** 3 classifiers (Logistic Regression, Random Forest, DNN) + 3 regressors (Lasso+Poly, Random Forest, DNN), all evaluated with 5-fold GroupKFold CV (grouped by athlete so no athlete leaks between train/test).

## Dataset

From Zenodo: [Synthetic Triathlete Dataset for Injury Prediction Research](https://zenodo.org/records/15401061) — Rossi, University of St. Gallen, 2024.

| File | Rows | What it is |
|---|---|---|
| `athletes.csv` | 1,000 | Athlete profiles (age, gender, VO2max, FTP, etc.) |
| `daily_data.csv` | 366,000 | Daily wearable data (HRV, RHR, sleep, stress, body battery) |
| `activity_data.csv` | 384,153 | Training sessions (TSS, HR zones, power zones) |

Data auto-downloads from Zenodo on first run.

## Setup

```bash
git clone https://github.com/felifire1/TriML.git
cd TriML
pip install -r requirements.txt
```

Needs Python 3.9+, PyTorch, scikit-learn, pandas, numpy, matplotlib, seaborn.

## How to reproduce

### Option 1: Colab (recommended)

Easiest way — runs on a free T4 GPU, takes ~20-30 min:

1. Open [`TriML_Colab.ipynb`](TriML_Colab.ipynb) in Google Colab
2. Set runtime to GPU (Runtime -> Change runtime type -> T4)
3. Run all cells
4. Download results zip when done

### Option 2: Run locally

```bash
# train all models (~30 min on CPU)
python3 ml_pipeline.py

# also run HP sweep
python3 ml_pipeline.py --tune

# generate plots
python3 generate_plots.py

# optional: streamlit dashboard
streamlit run app/streamlit_app.py
```

## Results

### Injury Classification (binary)
| Model | ROC-AUC | F1-macro | Accuracy |
|---|---|---|---|
| Logistic Regression | 0.875 | 0.699 | 85.8% |
| Random Forest | 0.901 | 0.839 | 95.1% |
| **DNN (MLP)** | **0.948** | **0.871** | **95.5%** |

### Load Classification (3-class)
| Model | ROC-AUC | F1-macro | Accuracy |
|---|---|---|---|
| Logistic Regression | 0.988 | 0.914 | 91.3% |
| Random Forest | 0.987 | 0.914 | 91.3% |
| **DNN (MLP)** | **0.997** | **0.959** | **95.9%** |

### Grit Score Regression
| Model | RMSE | R² |
|---|---|---|
| Lasso + Poly | 1.81 | 0.973 |
| Random Forest | 1.54 | 0.981 |
| **DNN (MLP)** | **1.19** | **0.988** |

## File structure

```
TriML/
  ml_pipeline.py          # main pipeline
  generate_plots.py       # plotting
  TriML_Colab.ipynb       # colab notebook
  requirements.txt
  src/
    loader.py             # data loading/parsing
    features.py           # feature engineering + grit score
    models.py             # model defs + CV
  app/
    streamlit_app.py      # dashboard
  results/
    ml_results.pkl
    hp_sweep.pkl
    plots/                # 12 PNGs
```

## References

1. Rossi, L. (2025). Synthetic Triathlete Dataset for Injury Prediction Research. Zenodo. https://doi.org/10.5281/zenodo.15401061
2. Gabbett, T.J. (2016). The training-injury prevention paradox. BJSM, 50(5), 273-280.
3. Hulin, B.T., et al. (2016). Spikes in acute workload are associated with increased injury risk. BJSM, 48(8), 708-712.
