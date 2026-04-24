# models.py - ML models for TriML project
# CS 6140 - Spring 2026
#
# 3 classifiers (LR, RF, MLP) and 3 regressors (Lasso+Poly, RF, MLP)
# using GroupKFold CV grouped by athlete_id

import time
import warnings
from typing import Any, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

N_FOLDS = 5
RANDOM_STATE = 42


###############################################################################
# MLP definition
###############################################################################

class MLP(nn.Module):
    """3 hidden layer MLP with batchnorm + dropout."""

    def __init__(self, in_features, out_features, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden // 2, hidden // 4),
            nn.BatchNorm1d(hidden // 4),
            nn.ReLU(),

            nn.Linear(hidden // 4, out_features),
        )

    def forward(self, x):
        return self.net(x)


def _train_mlp(X_train, y_train, X_val, task, n_classes=1,
               hidden=128, epochs=30, batch_size=1024, lr=1e-3):
    """Train an MLP for the given task. Returns (model, scaler, X_val_scaled, device)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train).astype(np.float32)
    X_vl = scaler.transform(X_val).astype(np.float32)

    out_dim = 1 if task in ("binary", "regression") else n_classes
    model = MLP(X_tr.shape[1], out_dim, hidden=hidden).to(device)

    # handle class imbalance for binary
    if task == "binary":
        pos_weight = torch.tensor(
            [(y_train == 0).sum() / max((y_train == 1).sum(), 1)],
            dtype=torch.float32,
        ).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif task == "multiclass":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    Xt = torch.from_numpy(X_tr)
    if task in ("binary", "regression"):
        yt = torch.from_numpy(y_train.astype(np.float32)).unsqueeze(1)
    else:
        yt = torch.from_numpy(y_train.astype(np.int64))

    ds = TensorDataset(Xt, yt)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            opt.step()
        scheduler.step()

    model.eval()
    return model, scaler, X_vl, device


def _predict_mlp(model, X_val_scaled, device, task, n_classes=1):
    """Get predictions from trained MLP."""
    with torch.no_grad():
        Xv = torch.from_numpy(X_val_scaled).to(device)
        logits = model(Xv).cpu().numpy()

    if task == "binary":
        proba = 1 / (1 + np.exp(-logits.squeeze()))
        pred = (proba >= 0.5).astype(int)
        return pred, proba
    elif task == "multiclass":
        from scipy.special import softmax
        proba = softmax(logits, axis=1)
        pred = proba.argmax(axis=1)
        return pred, proba
    else:  # regression
        return logits.squeeze(), None


###############################################################################
# CV runners
###############################################################################

def _clf_metrics(y_true, y_pred, y_proba, n_classes):
    """Compute classification metrics for one fold."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    try:
        if n_classes == 2:
            auc = roc_auc_score(y_true, y_proba)
        else:
            auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except Exception:
        auc = np.nan
    return {"accuracy": acc, "f1_macro": f1, "precision": prec, "recall": rec, "roc_auc": auc}


def _reg_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}


def run_classification_cv(X, y, groups, n_classes=2, label="injury"):
    """
    Train LR, RF, and MLP classifiers with GroupKFold CV.
    Returns dict with fold metrics, mean, std, and RF feature importances.
    """
    gkf = GroupKFold(n_splits=N_FOLDS)
    task = "binary" if n_classes == 2 else "multiclass"

    results = {name: {"fold_metrics": []} for name in ("lr", "rf", "mlp")}
    rf_importances = []

    scaler_global = StandardScaler()
    X_scaled = scaler_global.fit_transform(X)

    for fold, (tr_idx, vl_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_vl = X_scaled[tr_idx], X_scaled[vl_idx]
        y_tr, y_vl = y[tr_idx], y[vl_idx]

        # --- Logistic Regression ---
        lr = LogisticRegression(
            max_iter=1000, class_weight="balanced",
            random_state=RANDOM_STATE, C=0.5,
        )
        lr.fit(X_tr, y_tr)
        y_pred_lr = lr.predict(X_vl)
        y_prob_lr = lr.predict_proba(X_vl) if n_classes > 2 else lr.predict_proba(X_vl)[:, 1]
        results["lr"]["fold_metrics"].append(
            _clf_metrics(y_vl, y_pred_lr, y_prob_lr, n_classes)
        )

        # --- Random Forest ---
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=12,
            class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        )
        rf.fit(X_tr, y_tr)
        y_pred_rf = rf.predict(X_vl)
        y_prob_rf = rf.predict_proba(X_vl) if n_classes > 2 else rf.predict_proba(X_vl)[:, 1]
        results["rf"]["fold_metrics"].append(
            _clf_metrics(y_vl, y_pred_rf, y_prob_rf, n_classes)
        )
        rf_importances.append(rf.feature_importances_)

        # --- MLP (PyTorch) ---
        model, _, X_vl_sc, device = _train_mlp(
            X[tr_idx], y_tr, X[vl_idx], task=task, n_classes=n_classes
        )
        y_pred_mlp, y_prob_mlp = _predict_mlp(model, X_vl_sc, device, task, n_classes)
        results["mlp"]["fold_metrics"].append(
            _clf_metrics(y_vl, y_pred_mlp, y_prob_mlp, n_classes)
        )

    # average across folds
    for name in ("lr", "rf", "mlp"):
        folds = results[name]["fold_metrics"]
        keys = folds[0].keys()
        results[name]["mean"] = {k: np.mean([f[k] for f in folds]) for k in keys}
        results[name]["std"] = {k: np.std([f[k] for f in folds]) for k in keys}

    results["feature_importance"] = np.mean(rf_importances, axis=0)
    return results


def run_regression_cv(X, y, groups, label="grit_score"):
    """
    Train Lasso+Poly, RF, and MLP regressors with GroupKFold CV.
    Returns dict with fold metrics, mean, std, and RF feature importances.
    """
    gkf = GroupKFold(n_splits=N_FOLDS)

    results = {name: {"fold_metrics": []} for name in ("lasso", "rf", "mlp")}
    rf_importances = []

    for fold, (tr_idx, vl_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_vl = X[tr_idx], X[vl_idx]
        y_tr, y_vl = y[tr_idx], y[vl_idx]

        # --- Lasso with polynomial interaction features ---
        lasso_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
            ("lasso", Lasso(alpha=0.01, max_iter=5000)),
        ])
        lasso_pipe.fit(X_tr, y_tr)
        y_pred_lasso = lasso_pipe.predict(X_vl)
        results["lasso"]["fold_metrics"].append(_reg_metrics(y_vl, y_pred_lasso))

        # --- Random Forest ---
        rf = RandomForestRegressor(
            n_estimators=200, max_depth=12,
            random_state=RANDOM_STATE, n_jobs=-1,
        )
        rf.fit(X_tr, y_tr)
        y_pred_rf = rf.predict(X_vl)
        results["rf"]["fold_metrics"].append(_reg_metrics(y_vl, y_pred_rf))
        rf_importances.append(rf.feature_importances_)

        # --- MLP ---
        model, _, X_vl_sc, device = _train_mlp(
            X_tr, y_tr, X_vl, task="regression"
        )
        y_pred_mlp, _ = _predict_mlp(model, X_vl_sc, device, "regression")
        results["mlp"]["fold_metrics"].append(_reg_metrics(y_vl, y_pred_mlp))

    # average across folds
    for name in ("lasso", "rf", "mlp"):
        folds = results[name]["fold_metrics"]
        keys = folds[0].keys()
        results[name]["mean"] = {k: np.mean([f[k] for f in folds]) for k in keys}
        results[name]["std"] = {k: np.std([f[k] for f in folds]) for k in keys}

    results["feature_importance"] = np.mean(rf_importances, axis=0)
    return results


###############################################################################
# Run everything
###############################################################################

def run_all_models(X, y_injury, y_grit, y_load, groups, feature_names):
    """Run all 9 models (3 clf tasks x 3 models each) and return results."""

    print("Running injury classification CV...", flush=True)
    injury_clf = run_classification_cv(X, y_injury, groups, n_classes=2, label="injury")

    print("Running load-class classification CV...", flush=True)
    load_clf = run_classification_cv(X, y_load, groups, n_classes=3, label="load_class")

    print("Running Grit Score regression CV...", flush=True)
    grit_reg = run_regression_cv(X, y_grit, groups, label="grit_score")

    return {
        "injury_clf": injury_clf,
        "load_clf": load_clf,
        "grit_reg": grit_reg,
        "feature_names": feature_names,
    }


def results_to_dataframes(results):
    """Convert raw CV results into nice DataFrames for printing/display."""

    def _clf_df(res, model_labels):
        rows = []
        for key, label in model_labels:
            m = res[key]["mean"]
            s = res[key]["std"]
            rows.append({
                "Model": label,
                "ROC-AUC":  f"{m['roc_auc']:.3f} ± {s['roc_auc']:.3f}",
                "F1-macro": f"{m['f1_macro']:.3f} ± {s['f1_macro']:.3f}",
                "Precision": f"{m['precision']:.3f} ± {s['precision']:.3f}",
                "Recall":   f"{m['recall']:.3f} ± {s['recall']:.3f}",
                "Accuracy": f"{m['accuracy']:.3f} ± {s['accuracy']:.3f}",
                "roc_auc_val": m["roc_auc"],
            })
        return pd.DataFrame(rows).sort_values("roc_auc_val", ascending=False).drop(columns="roc_auc_val")

    def _reg_df(res, model_labels):
        rows = []
        for key, label in model_labels:
            m = res[key]["mean"]
            s = res[key]["std"]
            rows.append({
                "Model": label,
                "RMSE": f"{m['rmse']:.3f} ± {s['rmse']:.3f}",
                "MAE":  f"{m['mae']:.3f} ± {s['mae']:.3f}",
                "R²":   f"{m['r2']:.3f} ± {s['r2']:.3f}",
                "r2_val": m["r2"],
            })
        return pd.DataFrame(rows).sort_values("r2_val", ascending=False).drop(columns="r2_val")

    def _fi_df(res, feature_names, top_n=15):
        fi = pd.Series(res["feature_importance"], index=feature_names)
        return fi.nlargest(top_n).reset_index().rename(columns={"index": "Feature", 0: "Importance"})

    clf_labels = [("lr", "Logistic Regression"), ("rf", "Random Forest"), ("mlp", "DNN (MLP)")]
    reg_labels = [("lasso", "Lasso + Poly"), ("rf", "Random Forest"), ("mlp", "DNN (MLP)")]

    return {
        "clf_injury": _clf_df(results["injury_clf"], clf_labels),
        "clf_load":   _clf_df(results["load_clf"], clf_labels),
        "reg_grit":   _reg_df(results["grit_reg"], reg_labels),
        "fi_injury":  _fi_df(results["injury_clf"], results["feature_names"]),
        "fi_grit":    _fi_df(results["grit_reg"], results["feature_names"]),
    }


###############################################################################
# Hyperparameter sweep
# (course requirement: investigate effect of hyperparameters via CV)
###############################################################################

def hyperparameter_sweep(X, y_clf, y_reg, groups, n_classes=3):
    """
    Sweep one key HP per model, report mean CV metric vs HP value.
    Uses 3-fold GroupKFold to keep it fast.
    """
    gkf3 = GroupKFold(n_splits=3)
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    task_clf = "binary" if n_classes == 2 else "multiclass"

    # helper: classification CV score for sklearn models
    def _cv_score_clf(model_fn):
        scores = []
        for tr, vl in gkf3.split(X, y_clf, groups):
            m = model_fn()
            m.fit(X_sc[tr], y_clf[tr])
            proba = m.predict_proba(X_sc[vl])
            if n_classes == 2:
                proba = proba[:, 1]
            try:
                s = roc_auc_score(y_clf[vl], proba,
                                  multi_class="ovr" if n_classes > 2 else "raise",
                                  average="macro" if n_classes > 2 else None)
            except Exception:
                s = np.nan
            scores.append(s)
        return np.nanmean(scores), np.nanstd(scores)

    # helper: regression CV score for sklearn models
    def _cv_score_reg(model_fn):
        scores = []
        for tr, vl in gkf3.split(X, y_reg, groups):
            m = model_fn()
            m.fit(X[tr], y_reg[tr])
            pred = m.predict(X[vl])
            scores.append(r2_score(y_reg[vl], pred))
        return np.nanmean(scores), np.nanstd(scores)

    # lasso needs its own pipeline
    def _cv_score_lasso(alpha):
        scores = []
        for tr, vl in gkf3.split(X, y_reg, groups):
            pipe = Pipeline([
                ("sc", StandardScaler()),
                ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
                ("lasso", Lasso(alpha=alpha, max_iter=5000)),
            ])
            pipe.fit(X[tr], y_reg[tr])
            scores.append(r2_score(y_reg[vl], pipe.predict(X[vl])))
        return np.nanmean(scores), np.nanstd(scores)

    def _cv_score_mlp_clf(hidden):
        scores = []
        for tr, vl in gkf3.split(X, y_clf, groups):
            model, _, X_vl_sc, device = _train_mlp(
                X[tr], y_clf[tr], X[vl], task=task_clf,
                n_classes=n_classes, hidden=hidden, epochs=20,
            )
            _, proba = _predict_mlp(model, X_vl_sc, device, task_clf, n_classes)
            if n_classes == 2:
                try: s = roc_auc_score(y_clf[vl], proba)
                except: s = np.nan
            else:
                try: s = roc_auc_score(y_clf[vl], proba, multi_class="ovr", average="macro")
                except: s = np.nan
            scores.append(s)
        return np.nanmean(scores), np.nanstd(scores)

    def _cv_score_mlp_reg(hidden):
        scores = []
        for tr, vl in gkf3.split(X, y_reg, groups):
            model, _, X_vl_sc, device = _train_mlp(
                X[tr], y_reg[tr], X[vl], task="regression",
                hidden=hidden, epochs=20,
            )
            pred, _ = _predict_mlp(model, X_vl_sc, device, "regression")
            scores.append(r2_score(y_reg[vl], pred))
        return np.nanmean(scores), np.nanstd(scores)

    results = {}

    # LR: regularization strength C
    print("  HP sweep: LR C...", flush=True)
    C_vals = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
    rows = []
    for c in C_vals:
        mu, sd = _cv_score_clf(
            lambda c=c: LogisticRegression(C=c, max_iter=1000,
                                           class_weight="balanced",
                                           random_state=RANDOM_STATE)
        )
        rows.append({"C": c, "mean_auc": mu, "std_auc": sd})
    results["lr_C"] = pd.DataFrame(rows)

    # RF classifier: max_depth
    print("  HP sweep: RF classifier max_depth...", flush=True)
    depths = [3, 5, 8, 12, 16, 20]
    rows = []
    for d in depths:
        mu, sd = _cv_score_clf(
            lambda d=d: RandomForestClassifier(n_estimators=100, max_depth=d,
                                               class_weight="balanced",
                                               random_state=RANDOM_STATE, n_jobs=-1)
        )
        rows.append({"max_depth": d, "mean_auc": mu, "std_auc": sd})
    results["rf_clf_depth"] = pd.DataFrame(rows)

    # DNN classifier: hidden layer size
    print("  HP sweep: DNN classifier hidden size...", flush=True)
    hidden_sizes = [32, 64, 128, 256]
    rows = []
    for h in hidden_sizes:
        mu, sd = _cv_score_mlp_clf(h)
        rows.append({"hidden_size": h, "mean_auc": mu, "std_auc": sd})
    results["dnn_clf_hidden"] = pd.DataFrame(rows)

    # Lasso: alpha
    print("  HP sweep: Lasso alpha...", flush=True)
    alphas = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0]
    rows = []
    for a in alphas:
        mu, sd = _cv_score_lasso(a)
        rows.append({"alpha": a, "mean_r2": mu, "std_r2": sd})
    results["lasso_alpha"] = pd.DataFrame(rows)

    # RF regressor: max_depth
    print("  HP sweep: RF regressor max_depth...", flush=True)
    rows = []
    for d in depths:
        mu, sd = _cv_score_reg(
            lambda d=d: RandomForestRegressor(n_estimators=100, max_depth=d,
                                              random_state=RANDOM_STATE, n_jobs=-1)
        )
        rows.append({"max_depth": d, "mean_r2": mu, "std_r2": sd})
    results["rf_reg_depth"] = pd.DataFrame(rows)

    # DNN regressor: hidden layer size
    print("  HP sweep: DNN regressor hidden size...", flush=True)
    rows = []
    for h in hidden_sizes:
        mu, sd = _cv_score_mlp_reg(h)
        rows.append({"hidden_size": h, "mean_r2": mu, "std_r2": sd})
    results["dnn_reg_hidden"] = pd.DataFrame(rows)

    return results
