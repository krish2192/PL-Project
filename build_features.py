#!/usr/bin/env python3
"""
Feature engineering for EPL match outcome prediction.

Input:  data/processed/epl_matches.parquet (or .csv)
Output: data/processed/epl_features.parquet + .csv

What we create (pre-match features):
- Rolling team form over last N matches (goals for/against, shots, shots on target, points)
- Days since last match and rest-difference
- Simple Elo ratings (pre-match) with home-advantage term
- Betting odds transformed to implied probabilities (if available)

Leakage precautions:
- All rolling stats and Elo are computed using ONLY matches strictly before the current match (shifted by 1).
- Sorting by date is mandatory.

Run:
  pip install -U pandas numpy pyarrow
  python build_features.py --input data/processed/epl_matches.parquet --out data/processed/epl_features.parquet
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class Config:
    rolling_windows: Tuple[int, ...] = (3, 5, 10)
    elo_k: float = 20.0
    elo_home_adv: float = 50.0  # points added to home team pre-match


POINTS_MAP = {"H": (3, 0), "A": (0, 3), "D": (1, 1)}  # (home_points, away_points)

# Fallback mapping if the input is a raw football-data.co.uk CSV (not normalized by ingest)
RAW_TO_CANON = {
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "full_time_home_goals",
    "FTAG": "full_time_away_goals",
    "FTR": "full_time_result",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",
    "HC": "home_corners",
    "AC": "away_corners",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HY": "home_yellow",
    "AY": "away_yellow",
    "HR": "home_red",
    "AR": "away_red",
    "B365H": "b365_home_odds",
    "B365D": "b365_draw_odds",
    "B365A": "b365_away_odds",
}


def load_matches(path: Path) -> pd.DataFrame:
    """Load matches from CSV or Parquet and ensure canonical column names exist.
    - Auto-detect common date column names ("date" or "Date") and parse them.
    - If the file is a raw football-data.co.uk CSV, map raw headers -> canonical.
    """
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, low_memory=False)
    else:
        df = pd.read_parquet(path)

    # --- Date column detection & parsing ---
    date_col = None
    for c in df.columns:
        if str(c).lower() in ("date", "match_date"):
            date_col = c
            break
    if date_col is None:
        raise ValueError(
            "No date-like column found. Expected one of ['date','Date','match_date']."
            f"Found columns: {list(df.columns)[:20]} ..."
        )
    if date_col != "date":
        df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True, infer_datetime_format=True)

    # --- Map raw football-data headers to our canonical names if needed ---
    # Only rename if canonical doesn't already exist
    for raw, canon in RAW_TO_CANON.items():
        if raw in df.columns and canon not in df.columns:
            df = df.rename(columns={raw: canon})

    # basic safety filters
    required = ["home_team", "away_team"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns {missing}. If you're passing raw E0.csv files, "
            f"please either run data_ingest.py first or keep RAW_TO_CANON mapping updated."
        )

    df = df.dropna(subset=["date", "home_team", "away_team"]).copy()
    df = df.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    return df


def add_points(df: pd.DataFrame) -> pd.DataFrame:
    # derive points earned from FTR
    hp, ap = [], []
    for r in df.get("full_time_result", pd.Series([np.nan]*len(df))):
        if r in POINTS_MAP:
            h, a = POINTS_MAP[r]
        else:
            h, a = (np.nan, np.nan)
        hp.append(h); ap.append(a)
    df["home_points"] = hp
    df["away_points"] = ap
    return df


def long_format_team_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Create a long dataframe keyed by (team, date) with per-match stats and outcome.
    We'll use it to compute rolling features, then pivot back to match-level.
    """
    # home rows
    home_cols = {
        "team": df["home_team"],
        "opponent": df["away_team"],
        "venue": "home",
        "date": df["date"],
        "goals_for": df.get("full_time_home_goals"),
        "goals_against": df.get("full_time_away_goals"),
        "shots_for": df.get("home_shots"),
        "shots_against": df.get("away_shots"),
        "sot_for": df.get("home_shots_on_target"),
        "sot_against": df.get("away_shots_on_target"),
        "points": df.get("home_points"),
    }
    home = pd.DataFrame(home_cols)

    # away rows
    away_cols = {
        "team": df["away_team"],
        "opponent": df["home_team"],
        "venue": "away",
        "date": df["date"],
        "goals_for": df.get("full_time_away_goals"),
        "goals_against": df.get("full_time_home_goals"),
        "shots_for": df.get("away_shots"),
        "shots_against": df.get("home_shots"),
        "sot_for": df.get("away_shots_on_target"),
        "sot_against": df.get("home_shots_on_target"),
        "points": df.get("away_points"),
    }
    away = pd.DataFrame(away_cols)

    long_df = pd.concat([home, away], ignore_index=True)
    # ensure numeric columns are numeric
    for c in [
        "goals_for", "goals_against", "shots_for", "shots_against", "sot_for", "sot_against", "points",
    ]:
        if c in long_df:
            long_df[c] = pd.to_numeric(long_df[c], errors="coerce")
    long_df = long_df.sort_values(["team", "date"]).reset_index(drop=True)
    return long_df


def rolling_features(long_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    feats = long_df.copy()
    feats["days_since_last"] = (
        feats.groupby("team")["date"].diff().dt.days
    )

    for w in cfg.rolling_windows:
        grp = feats.groupby("team", group_keys=False)
        # shift(1) -> exclude current match from its own rolling window
        feats[f"gf_avg_{w}"] = grp["goals_for"].apply(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        feats[f"ga_avg_{w}"] = grp["goals_against"].apply(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        feats[f"s_for_avg_{w}"] = grp["shots_for"].apply(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        feats[f"s_against_avg_{w}"] = grp["shots_against"].apply(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        feats[f"sot_for_avg_{w}"] = grp["sot_for"].apply(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        feats[f"sot_against_avg_{w}"] = grp["sot_against"].apply(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        feats[f"pts_sum_{w}"] = grp["points"].apply(lambda s: s.shift(1).rolling(w, min_periods=1).sum())

    return feats


def compute_elo(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Compute simple Elo ratings pre-match for each team.
    We iterate matches chronologically and update after each match.
    Return a dataframe with columns [date, home_team, away_team, elo_home_pre, elo_away_pre].
    """
    elo = {}
    records = []

    def get(team: str) -> float:
        return elo.get(team, 1500.0)

    for _, row in df.sort_values("date").iterrows():
        h, a = row["home_team"], row["away_team"]
        r_home, r_away = get(h), get(a)
        # store pre-match
        records.append((row["date"], h, a, r_home, r_away))

        # expected scores (home adv)
        r_home_eff = r_home + cfg.elo_home_adv
        r_away_eff = r_away
        e_home = 1.0 / (1.0 + 10 ** (-(r_home_eff - r_away_eff) / 400))
        e_away = 1.0 - e_home

        # actual outcome
        res = row.get("full_time_result")
        if res == "H":
            s_home, s_away = 1.0, 0.0
        elif res == "A":
            s_home, s_away = 0.0, 1.0
        elif res == "D":
            s_home, s_away = 0.5, 0.5
        else:
            # unknown outcome -> skip update
            continue

        # updates
        elo[h] = r_home + cfg.elo_k * (s_home - e_home)
        elo[a] = r_away + cfg.elo_k * (s_away - e_away)

    elo_df = pd.DataFrame(records, columns=["date", "home_team", "away_team", "elo_home_pre", "elo_away_pre"])
    return elo_df


def implied_probs_from_odds(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if {"b365_home_odds", "b365_draw_odds", "b365_away_odds"}.issubset(out.columns):
        for col in ["b365_home_odds", "b365_draw_odds", "b365_away_odds"]:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        inv_sum = (1.0 / out["b365_home_odds"]) + (1.0 / out["b365_draw_odds"]) + (1.0 / out["b365_away_odds"])
        out["b365_p_home"] = (1.0 / out["b365_home_odds"]) / inv_sum
        out["b365_p_draw"] = (1.0 / out["b365_draw_odds"]) / inv_sum
        out["b365_p_away"] = (1.0 / out["b365_away_odds"]) / inv_sum
    return out


def build_dataset(matches_path: Path, out_path: Path, cfg: Config) -> pd.DataFrame:
    matches = load_matches(matches_path)
    matches = add_points(matches)

    # Elo pre-match
    elo_df = compute_elo(matches, cfg)
    m = matches.merge(elo_df, on=["date", "home_team", "away_team"], how="left")

    # Implied probabilities from odds (if present)
    m = implied_probs_from_odds(m)

    # Long format per-team rows for rolling features
    long_df = long_format_team_stats(m)
    long_feats = rolling_features(long_df, cfg)

    # For each match, we want the home team's pre-match rolling stats and the away team's
    # Merge by (team,date) shifted already inside rolling by using shift(1) before rolling.
    # Since they are aligned on the same date, we can merge on: home_team/date and away_team/date.

    # Home features
    home_merge_cols = ["team", "date", "days_since_last"] + [
        f for f in long_feats.columns if re_match_any(f, ["gf_avg_", "ga_avg_", "s_for_avg_", "s_against_avg_", "sot_for_avg_", "sot_against_avg_", "pts_sum_"])
    ]

    def prefix_cols(df: pd.DataFrame, prefix: str, ignore: List[str]) -> pd.DataFrame:
        ren = {c: f"{prefix}{c}" for c in df.columns if c not in ignore}
        return df.rename(columns=ren)

    home_feats = long_feats[home_merge_cols].rename(columns={"team": "home_team", "days_since_last": "home_days_since_last"})
    home_feats = prefix_cols(home_feats, "home_", ["home_team", "date", "home_days_since_last"])

    # Away features
    away_feats = long_feats[home_merge_cols].rename(columns={"team": "away_team", "days_since_last": "away_days_since_last"})
    away_feats = prefix_cols(away_feats, "away_", ["away_team", "date", "away_days_since_last"])

    feat = m.merge(home_feats, on=["home_team", "date"], how="left")
    feat = feat.merge(away_feats, on=["away_team", "date"], how="left")

    # Rest difference
    if {"home_days_since_last", "away_days_since_last"}.issubset(feat.columns):
        feat["rest_diff"] = feat["home_days_since_last"] - feat["away_days_since_last"]

    # Target encoding: y in {H,D,A}
    feat["y"] = feat["full_time_result"].map({"H": 0, "D": 1, "A": 2})

    # Keep only rows with known targets
    feat = feat.dropna(subset=["y"]).copy()
    feat["y"] = feat["y"].astype(int)

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".csv":
        feat.to_csv(out_path, index=False)
    else:
        feat.to_parquet(out_path, index=False)

    # also save a CSV twin for convenience
    twin_csv = out_path.with_suffix(".csv")
    feat.to_csv(twin_csv, index=False)
    return feat


def re_match_any(s: str, patterns: List[str]) -> bool:
    return any(s.startswith(p) for p in patterns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("data/processed/epl_matches.parquet"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/epl_features.parquet"))
    ap.add_argument("--elo-k", type=float, default=20.0)
    ap.add_argument("--elo-home-adv", type=float, default=50.0)
    args = ap.parse_args()

    cfg = Config(elo_k=args.elo_k, elo_home_adv=args.elo_home_adv)
    feat = build_dataset(args.input, args.out, cfg)

    # Minimal console summary
    print("Rows:", len(feat))
    print("Columns:", len(feat.columns))
    if {"date"}.issubset(feat.columns):
        print("Date span:", feat["date"].min().date(), "to", feat["date"].max().date())
    print("Target distribution (0=H,1=D,2=A):")
    print(feat["y"].value_counts().sort_index())


if __name__ == "__main__":
    main()
