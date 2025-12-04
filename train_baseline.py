#!/usr/bin/env python3
"""
Train a baseline classifier for EPL match outcomes (H/D/A) with time-aware validation.

Model 1 (baseline): Multinomial Logistic Regression
- Simple, fast, interpretable coefficients
- Good sanity check for leakage and feature usefulness

(Optional) Model 2: LightGBM multiclass (uncomment if installed)

Outputs:
- metrics printed to console (accuracy, log loss, macro F1)
- saved model to models/baseline_logreg.pkl

Run:
  pip install -U scikit-learn pandas numpy joblib
  python train_baseline.py --features data/processed/epl_features.parquet --model-out models/baseline_logreg.pkl
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler



TARGET = "y"
DATE_COL = "date"
ID_COLS = ["match_id", "home_team", "away_team", "season", DATE_COL]


# choose features: numeric + categorical
NUMERIC_CANDIDATES = [
    # odds (already transformed to implied probs if present)
    "b365_p_home", "b365_p_draw", "b365_p_away",
    # elo
    "elo_home_pre", "elo_away_pre",
    # rest diff
    "home_days_since_last", "away_days_since_last", "rest_diff",
]
# Rolling aggregates (pattern-based selection at runtime to keep it flexible)
ROLL_PREFIXES = (
    "home_gf_avg_", "home_ga_avg_", "home_s_for_avg_", "home_s_against_avg_", "home_sot_for_avg_", "home_sot_against_avg_", "home_pts_sum_",
    "away_gf_avg_", "away_ga_avg_", "away_s_for_avg_", "away_s_against_avg_", "away_sot_for_avg_", "away_sot_against_avg_", "away_pts_sum_",
)

CAT_CANDIDATES = ["home_team", "away_team", "season"]


def select_columns(df: pd.DataFrame):
    numeric_cols = [c for c in NUMERIC_CANDIDATES if c in df.columns]
    for c in df.columns:
        if c.startswith(ROLL_PREFIXES):
            numeric_cols.append(c)
    cat_cols = [c for c in CAT_CANDIDATES if c in df.columns]
    return numeric_cols, cat_cols


def load_features(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        feat = pd.read_csv(path, parse_dates=[DATE_COL])
    else:
        feat = pd.read_parquet(path)
    return feat


def build_pipeline(numeric_cols, cat_cols) -> Pipeline:
    # Pipelines with imputers handle NaNs from odds missingness, first-match rest days, early rolling windows, etc.
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler(with_mean=False)),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if cat_cols:
        transformers.append(("cat", categorical_pipeline, cat_cols))

    pre = ColumnTransformer(transformers=transformers, remainder="drop")

    clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=200,
        n_jobs=None,
    )

    pipe = Pipeline([
        ("pre", pre),
        ("clf", clf),
    ])
    return pipe



# Uncomment to try LightGBM (pip install lightgbm)
# from lightgbm import LGBMClassifier
# def build_lgbm_pipeline(numeric_cols, cat_cols) -> Pipeline:
#     pre = ColumnTransformer([
#         ("num", "passthrough", numeric_cols),
#         ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
#     ])
#     clf = LGBMClassifier(objective="multiclass", n_estimators=600, learning_rate=0.05, subsample=0.9)
#     return Pipeline([("pre", pre), ("clf", clf)])


def timeseries_cv_scores(df: pd.DataFrame, pipe: Pipeline, n_splits=5):
    # Sort chronologically to respect time
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
    ap.add_argument("--model-out", type=Path, default=Path("models/baseline_logreg.pkl"))
    ap.add_argument("--splits", type=int, default=5)
    args = ap.parse_args()

    args.model_out.parent.mkdir(parents=True, exist_ok=True)

    feat = load_features(args.features)

    # Drop any rows missing the target
    feat = feat.dropna(subset=[TARGET]).copy()

    numeric_cols, cat_cols = select_columns(feat)
    print("Numeric features:", len(numeric_cols))
    print("Categorical features:", cat_cols)

    pipe = build_pipeline(numeric_cols, cat_cols)

    # Time-series CV (no leakage)
    timeseries_cv_scores(feat, pipe, n_splits=args.splits)

    # Final fit on ALL data, then save model
    feat = feat.sort_values(DATE_COL).reset_index(drop=True)
    X_all = feat.drop(columns=[TARGET])
    y_all = feat[TARGET].values
    pipe.fit(X_all, y_all)

    dump(pipe, args.model_out)
    print("Saved model ->", args.model_out)


if __name__ == "__main__":
    main()
