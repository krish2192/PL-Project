#!/usr/bin/env python3
"""
Production-ready FastAPI backend for the EPL Predictor website.

What this service does
----------------------
- Loads your engineered matches dataset (data/processed/epl_matches.csv)
- Loads a trained model bundle (models/lgbm_calibrated.pkl preferred, else baseline_logreg.pkl)
- Exposes:
  * GET  /health        -> quick status
  * GET  /teams         -> list of known team names
  * POST /predict       -> probability for Home/Draw/Away for a (home_team, away_team, date)

How it works
------------
For a requested match (home, away, date), we:
1) Filter historical matches strictly earlier than `date`
2) Compute pre-match features for both teams (rolling form, shots, points; Elo up to date; rest_diff)
3) Build a single-row feature frame matching training schema
4) Predict probabilities with the trained pipeline; apply isotonic calibration if available

Run locally
-----------
# 1) Install deps
#    pip install -U fastapi uvicorn[standard] pandas numpy scikit-learn joblib lightgbm
# 2) Start the server
#    uvicorn api_server:app --reload --port 8000
# 3) Try it
#    curl "http://127.0.0.1:8000/teams"
#    curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" \
#         -d '{"home_team":"Arsenal","away_team":"Chelsea","date":"2025-08-12"}'

Notes
-----
- This API recomputes features on-the-fly from the historical matches table. For production,
  you can cache per-team rolling snapshots by date to speed up cold requests.
- OneHotEncoder(handle_unknown="ignore") in your training pipeline means unseen seasons/teams won't crash.
"""
from __future__ import annotations

import math
from datetime import datetime, date as date_cls
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# -----------------------
# Config paths
# -----------------------
MATCHES_PATH = Path("data/processed/epl_matches.csv")  # normalized by data_ingest.py
MODEL_BUNDLE_PREFERRED = Path("models/lgbm_calibrated.pkl")
MODEL_FALLBACK = Path("models/baseline_logreg.pkl")

# -----------------------
# FastAPI setup
# -----------------------
app = FastAPI(title="EPL Predictor API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def ensure_expected_columns(X: pd.DataFrame, pipe) -> pd.DataFrame:
    """
    Make sure all columns the pipeline's preprocessor expects are present in X.
    Missing ones are added as NaN so imputers / encoders can handle them.
    """
    try:
        pre = pipe.named_steps.get("pre")
    except Exception:
        pre = None
    if pre is None:
        return X

    expected = []
    for name, trans, cols in pre.transformers_:
        if cols is None or cols == "drop":
            continue
        # cols is usually a list of column names
        if isinstance(cols, (list, tuple)):
            expected.extend(list(cols))
    # Add missing columns with NaN
    for c in expected:
        if c not in X.columns:
            X[c] = np.nan
    return X


# -----------------------
# Data loading
# -----------------------
if not MATCHES_PATH.exists():
    raise RuntimeError(f"Matches dataset not found at {MATCHES_PATH}. Run data_ingest.py first.")

matches = pd.read_csv(MATCHES_PATH, parse_dates=["date"])
# Ensure expected columns exist (ingest sets these names)
_required = {"date","home_team","away_team","full_time_result"}
if not _required.issubset(matches.columns):
    raise RuntimeError(f"Matches CSV missing columns: {_required - set(matches.columns)}")

# dedupe and sort
matches = matches.drop_duplicates().sort_values("date").reset_index(drop=True)

# quick set of teams for UI
TEAMS: List[str] = sorted(pd.unique(pd.concat([matches["home_team"], matches["away_team"]], ignore_index=True)))

# -----------------------
# Model loading
# -----------------------
MODEL_BUNDLE = None
PIPELINE_ONLY = None
if MODEL_BUNDLE_PREFERRED.exists():
    # advanced_modeling.py saves a dict: {"pipeline": pipe, "calibrator": calib, "classes": [0,1,2]}
    MODEL_BUNDLE = joblib.load(MODEL_BUNDLE_PREFERRED)
elif MODEL_FALLBACK.exists():
    PIPELINE_ONLY = joblib.load(MODEL_FALLBACK)
else:
    raise RuntimeError("No trained model found. Train with train_lgbm.py or train_baseline.py.")

# Allowed date window (as-of cutoff)
MIN_ALLOWED = (matches["date"].min().normalize() + pd.Timedelta(days=1))
MAX_ALLOWED = matches["date"].max().normalize()

@app.get("/limits")
def limits():
    return {
        "min": MIN_ALLOWED.strftime("%Y-%m-%d"),
        "max": MAX_ALLOWED.strftime("%Y-%m-%d"),
        "default": MAX_ALLOWED.strftime("%Y-%m-%d"),
}

# -----------------------
# Pydantic schemas
# -----------------------
class PredictIn(BaseModel):
    home_team: str = Field(..., description="Home team name")
    away_team: str = Field(..., description="Away team name")
    date: datetime | date_cls = Field(..., description="Match date (YYYY-MM-DD)")

class PredictOut(BaseModel):
    home_win: float
    draw: float
    away_win: float
    top_class: str

# -----------------------
# Feature helpers (re-implement minimal pieces from build_features)
# -----------------------
ROLLING_WINDOWS = (3, 5, 10)
ELO_K = 20.0
ELO_HOME_ADV = 50.0

POINTS_MAP = {"H": (3, 0), "D": (1, 1), "A": (0, 3)}


def add_points(df: pd.DataFrame) -> pd.DataFrame:
    if "full_time_result" not in df:
        return df
    hp, ap = [], []
    for r in df["full_time_result"].astype(str):
        h, a = POINTS_MAP.get(r, (np.nan, np.nan))
        hp.append(h); ap.append(a)
    df = df.copy()
    df["home_points"], df["away_points"] = hp, ap
    return df


def long_format(df: pd.DataFrame) -> pd.DataFrame:
    df = add_points(df)
    home = pd.DataFrame({
        "team": df["home_team"],
        "opponent": df["away_team"],
        "date": df["date"],
        "goals_for": df.get("full_time_home_goals"),
        "goals_against": df.get("full_time_away_goals"),
        "shots_for": df.get("home_shots"),
        "shots_against": df.get("away_shots"),
        "sot_for": df.get("home_shots_on_target"),
        "sot_against": df.get("away_shots_on_target"),
        "points": df.get("home_points"),
        "venue": "home",
    })
    away = pd.DataFrame({
        "team": df["away_team"],
        "opponent": df["home_team"],
        "date": df["date"],
        "goals_for": df.get("full_time_away_goals"),
        "goals_against": df.get("full_time_home_goals"),
        "shots_for": df.get("away_shots"),
        "shots_against": df.get("home_shots"),
        "sot_for": df.get("away_shots_on_target"),
        "sot_against": df.get("home_shots_on_target"),
        "points": df.get("away_points"),
        "venue": "away",
    })
    long_df = pd.concat([home, away], ignore_index=True)
    for c in ["goals_for","goals_against","shots_for","shots_against","sot_for","sot_against","points"]:
        if c in long_df:
            long_df[c] = pd.to_numeric(long_df[c], errors="coerce")
    return long_df.sort_values(["team","date"]).reset_index(drop=True)


def rolling_snapshot(long_df: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    # filter strictly before as_of
    hist = long_df[long_df["date"] < as_of].copy()
    if hist.empty:
        return pd.DataFrame()
    hist["days_since_last"] = hist.groupby("team")["date"].diff().dt.days
    snaps = []
    for team, g in hist.groupby("team"):
        g = g.sort_values("date")
        row = {"team": team}
        # last rest
        row["days_since_last"] = (g["date"].iloc[-1] - g["date"].iloc[-2]).days if len(g) >= 2 else np.nan
        for w in ROLLING_WINDOWS:
            s = g.iloc[:-0]  # just clarity
            row[f"gf_avg_{w}"] = g["goals_for"].rolling(w, min_periods=1).mean().iloc[-1]
            row[f"ga_avg_{w}"] = g["goals_against"].rolling(w, min_periods=1).mean().iloc[-1]
            row[f"s_for_avg_{w}"] = g["shots_for"].rolling(w, min_periods=1).mean().iloc[-1]
            row[f"s_against_avg_{w}"] = g["shots_against"].rolling(w, min_periods=1).mean().iloc[-1]
            row[f"sot_for_avg_{w}"] = g["sot_for"].rolling(w, min_periods=1).mean().iloc[-1]
            row[f"sot_against_avg_{w}"] = g["sot_against"].rolling(w, min_periods=1).mean().iloc[-1]
            row[f"pts_sum_{w}"] = g["points"].rolling(w, min_periods=1).sum().iloc[-1]
        snaps.append(row)
    return pd.DataFrame(snaps)


def elo_until(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    elo = {}
    records = []
    prior = df[df["date"] < as_of].sort_values("date")
    for _, r in prior.iterrows():
        h, a = r["home_team"], r["away_team"]
        rh, ra = elo.get(h, 1500.0), elo.get(a, 1500.0)
        records.append((h, rh))
        records.append((a, ra))
        # expected
        eh = 1.0 / (1.0 + 10 ** (-((rh + ELO_HOME_ADV) - ra) / 400))
        if r.get("full_time_result") == "H": sh, sa = 1.0, 0.0
        elif r.get("full_time_result") == "A": sh, sa = 0.0, 1.0
        elif r.get("full_time_result") == "D": sh, sa = 0.5, 0.5
        else: continue
        elo[h] = rh + ELO_K * (sh - eh)
        elo[a] = ra + ELO_K * (sa - (1.0 - eh))
    # latest rating per team
    last = {}
    for t, r in records:
        last[t] = r
    return pd.DataFrame({"team": list(last.keys()), "elo_pre": list(last.values())})


def season_label(d: pd.Timestamp) -> str:
    y = d.year
    # EPL season roughly Aug -> May; if month >= July, season starts this year to next
    if d.month >= 7:
        return f"{y}-{y+1}"
    else:
        return f"{y-1}-{y}"


def build_features_for_match(home: str, away: str, when: pd.Timestamp) -> pd.DataFrame:
    # subset historical matches strictly before 'when'
    hist = matches[matches["date"] < when].copy()
    if hist.empty:
        raise HTTPException(status_code=400, detail="No historical data before this date. Choose a later date.")

    # long form + rolling snapshots
    long_df = long_format(hist)
    snaps = rolling_snapshot(long_df, when)

    # join for home / away
    home_row = snaps[snaps["team"] == home].copy()
    away_row = snaps[snaps["team"] == away].copy()

    # elo
    elo_df = elo_until(hist, when)
    h_elo = float(elo_df.loc[elo_df["team"] == home, "elo_pre"].iloc[0]) if (elo_df["team"] == home).any() else np.nan
    a_elo = float(elo_df.loc[elo_df["team"] == away, "elo_pre"].iloc[0]) if (elo_df["team"] == away).any() else np.nan

    # assemble one-row features
    row = {
        "date": when,
        "home_team": home,
        "away_team": away,
        "season": season_label(when),
        "elo_home_pre": h_elo,
        "elo_away_pre": a_elo,
    }

    if not home_row.empty:
        row.update({
            "home_days_since_last": home_row["days_since_last"].iloc[0],
            **{f"home_{c}": home_row[c].iloc[0] for c in home_row.columns if c.startswith(("gf_avg_","ga_avg_","s_for_avg_","s_against_avg_","sot_for_avg_","sot_against_avg_","pts_sum_"))}
        })
    if not away_row.empty:
        row.update({
            "away_days_since_last": away_row["days_since_last"].iloc[0],
            **{f"away_{c}": away_row[c].iloc[0] for c in away_row.columns if c.startswith(("gf_avg_","ga_avg_","s_for_avg_","s_against_avg_","sot_for_avg_","sot_against_avg_","pts_sum_"))}
        })

    if "home_days_since_last" in row and "away_days_since_last" in row:
        row["rest_diff"] = row["home_days_since_last"] - row["away_days_since_last"]

    return pd.DataFrame([row])


# -----------------------
# Routes
# -----------------------
@app.get("/health")
def health():
    return {"ok": True, "matches": len(matches), "model": "lgbm_calibrated" if MODEL_BUNDLE else "baseline_logreg"}


@app.get("/teams", response_model=List[str])
def teams():
    return TEAMS


@app.post("/predict", response_model=PredictOut)
def predict(inp: PredictIn):
    home = inp.home_team
    away = inp.away_team
    when = pd.to_datetime(inp.date)
    # clamp to allowed window
    if not (MIN_ALLOWED <= when <= MAX_ALLOWED):
        raise HTTPException(
            status_code=400,
            detail=f"date must be between {MIN_ALLOWED.date()} and {MAX_ALLOWED.date()}",
    )


    if home not in TEAMS or away not in TEAMS:
        raise HTTPException(status_code=400, detail="Unknown team. Call /teams first to see valid names.")
    if home == away:
        raise HTTPException(status_code=400, detail="Home and away teams must be different.")

    X = build_features_for_match(home, away, when)

    if MODEL_BUNDLE is not None:
        pipe = MODEL_BUNDLE["pipeline"]
        X = ensure_expected_columns(X, pipe) 
        proba = pipe.predict_proba(X)
        proba = MODEL_BUNDLE["calibrator"].predict(proba)
    else:
        pipe = PIPELINE_ONLY
        X = ensure_expected_columns(X, pipe) 
        proba = pipe.predict_proba(X)

    # order 0,1,2 = Home, Draw, Away
    p_home, p_draw, p_away = map(float, proba[0].tolist())
    labels = ["Home", "Draw", "Away"]
    top_idx = int(np.argmax(proba[0]))
    return PredictOut(
        home_win=p_home,
        draw=p_draw,
        away_win=p_away,
        top_class=labels[top_idx],
    )
