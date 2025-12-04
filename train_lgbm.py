#!/usr/bin/env python3
"""
Train a stronger tabular model (LightGBM) for EPL match outcomes (H/D/A)
with time-aware cross‑validation and robust preprocessing.

Why LightGBM?
- Handles non-linearities and feature interactions better than logistic regression
- Works well on mixed numeric + categorical (via one‑hot here for simplicity)
- Tolerant to missing values (but we still impute explicitly)

Outputs:
- CV fold metrics (Accuracy, Log Loss, Macro‑F1)
- Saved model pipeline -> models/lgbm.pkl

Run:
  pip install -U lightgbm scikit-learn pandas numpy joblib
  python train_lgbm.py --features data/processed/epl_features.parquet --model-out models/lgbm.pkl --splits 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMClassifier

TARGET = "y"
DATE_COL = "date"
ID_COLS = ["match_id", "home_team", "away_team", "season", DATE_COL]

NUMERIC_PREFIXES = (
    "b365_p_", "elo_", "home_days_since_last", "away_days_since_last", "rest_diff",
    "home_gf_avg_", "home_ga_avg_", "home_s_for_avg_", "home_s_against_avg_",
    "home_sot_for_avg_", "home_sot_against_avg_", "home_pts_sum_",
    "away_gf_avg_", "away_ga_avg_", "away_s_for_avg_", "away_s_against_avg_",
    "away_sot_for_avg_", "away_sot_against_avg_", "away_pts_sum_",
)
CAT_COLS = ["home_team", "away_team", "season"]


def load_features(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        feat = pd.read_csv(path, low_memory=False, parse_dates=[DATE_COL])
    else:
        feat = pd.read_parquet(path)
    return feat


def select_columns(df: pd.DataFrame):
    numeric_cols = []
    for c in df.columns:
        if c == TARGET or c in ID_COLS:
            continue
        if any(c.startswith(p) for p in NUMERIC_PREFIXES):
            numeric_cols.append(c)
    cats = [c for c in CAT_COLS if c in df.columns]
    return numeric_cols, cats


def build_pipeline(numeric_cols, cat_cols) -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
        ]), numeric_cols),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_cols),
    ], remainder="drop")

    clf = LGBMClassifier(
        objective="multiclass",
        n_estimators=600,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        max_depth=-1,
        reg_lambda=1.0,
        random_state=42,
    )

    pipe = Pipeline([
        ("pre", pre),
        ("clf", clf),
    ])
    return pipe


def timeseries_cv_scores(df: pd.DataFrame, pipe: Pipeline, n_splits=5):
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    X = df.drop(columns=[TARGET])
    y = df[TARGET].values

    tscv = TimeSeriesSplit(n_splits=n_splits)

    accs, lls, f1s = [], [], []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)

        accs.append(accuracy_score(y_test, y_pred))
        lls.append(log_loss(y_test, y_proba))
        f1s.append(f1_score(y_test, y_pred, average="macro"))

        print(f"Fold {fold+1}: acc={accs[-1]:.3f}  logloss={lls[-1]:.3f}  macroF1={f1s[-1]:.3f}")

    print("CV mean ± std:")
    print(f"  Accuracy : {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"  Log Loss : {np.mean(lls):.3f} ± {np.std(lls):.3f}")
    print(f"  Macro F1 : {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=Path("data/processed/epl_features.parquet"))
    ap.add_argument("--model-out", type=Path, default=Path("models/lgbm.pkl"))
    ap.add_argument("--splits", type=int, default=5)
    args = ap.parse_args()

    args.model_out.parent.mkdir(parents=True, exist_ok=True)

    feat = load_features(args.features)
    feat = feat.dropna(subset=[TARGET]).copy()

    num_cols, cat_cols = select_columns(feat)
    print("Numeric features:", len(num_cols))
    print("Categorical features:", cat_cols)

    pipe = build_pipeline(num_cols, cat_cols)

    # Time-aware CV
    timeseries_cv_scores(feat, pipe, n_splits=args.splits)

    # Final fit on all data, then save
    feat = feat.sort_values(DATE_COL).reset_index(drop=True)
    X_all = feat.drop(columns=[TARGET])
    y_all = feat[TARGET].values
    pipe.fit(X_all, y_all)

    dump(pipe, args.model_out)
    print("Saved model ->", args.model_out)


if __name__ == "__main__":
    main()
