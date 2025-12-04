#!/usr/bin/env python3
"""
Data Ingestion & Cleaning for Premier League (football-data.co.uk)

What this script does
---------------------
1) Loads one or more season CSVs (e.g., data/raw/epl/1819/E0.csv … 2425/E0.csv)
2) Normalizes column names into a canonical schema
3) Adds a human-readable Season string (e.g., "2018-2019")
4) Parses dates robustly (day-first) and builds a stable match_id
5) Harmonizes common team-name aliases (e.g., "Man United" -> "Manchester United")
6) Concatenates all seasons and writes a single CSV + Parquet

How to run
----------
# (recommended) create a venv first, then install deps
# python -m venv .venv && source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
# pip install -U pandas pyarrow

# Example command (reads all E0.csv files under data/raw/epl/**/)
# python data_ingest.py --input-root data/raw/epl --pattern "E0.csv" --out-csv data/processed/epl_matches.csv --out-parquet data/processed/epl_matches.parquet

Notes
-----
- The football-data.co.uk schema is fairly consistent, but some fields can be missing per season.
- This script is defensive: it keeps any of the canonical columns that exist; missing ones are left as NaN.
- You can safely extend the TEAM_ALIASES dict below as you encounter new variations.
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd

# ----------------------------
# Logging setup
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger("ingest")

# ----------------------------
# Canonical schema (target columns)
# ----------------------------
CANONICAL_COLS = {
    "Date": "date",
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


# ----------------------------
# Team alias harmonization
# ----------------------------
TEAM_ALIASES: Dict[str, str] = {
    # Common shorthand to long form
    "Man United": "Manchester United",
    "Man City": "Manchester City",
    "Spurs": "Tottenham",
    "Wolves": "Wolverhampton",
    "West Brom": "West Bromwich Albion",
    "West Ham": "West Ham",
    "Newcastle": "Newcastle United",
    "Brighton": "Brighton",
    "Nott'm Forest": "Nottingham Forest",
    "Sheffield Utd": "Sheffield United",
    "Cardiff": "Cardiff City",
    "Swansea": "Swansea City",
    "Leeds": "Leeds United",
    "Leicester": "Leicester City",
    "Norwich": "Norwich City",
    "Huddersfield": "Huddersfield Town",
}

SEASON_CODE_RE = re.compile(r"(?P<s1>\\d{2})(?P<s2>\\d{2})")


def infer_season_from_path(path: Path) -> str | None:
    """Infer "YYYY-YYYY" from a parent folder like 1819, 2324, etc.
    Returns None if not found.
    """
    parts = list(path.parts)
    for p in reversed(parts):
        m = SEASON_CODE_RE.fullmatch(p)
        if m:
            y1 = int(m.group("s1"))
            y2 = int(m.group("s2"))
            # 00-29 -> 2000-2029; else -> 1900s (handles historical if needed)
            base1 = 2000 if y1 <= 29 else 1900
            base2 = 2000 if y2 <= 29 else 1900
            return f"{base1 + y1}-{base2 + y2}"
    return None


def canonicalize_team(name: str) -> str:
    if pd.isna(name):
        return name
    return TEAM_ALIASES.get(name, name)


def load_single_csv(csv_path: Path, season_hint: str | None) -> pd.DataFrame:
    logger.info(f"Loading {csv_path}")
    df = pd.read_csv(csv_path)

    # Add season column (prefer folder-inferred, else try from filename)
    season = season_hint or infer_season_from_path(csv_path.parent)
    if season is None:
        # try to sniff from anywhere in the path
        m = SEASON_CODE_RE.search(str(csv_path))
        if m:
            y1 = int(m.group("s1")); y2 = int(m.group("s2"))
            base1 = 2000 if y1 <= 29 else 1900
            base2 = 2000 if y2 <= 29 else 1900
            season = f"{base1 + y1}-{base2 + y2}"
    if season is None:
        season = "unknown"
    df["season"] = season

    keep_map = {raw: CANONICAL_COLS[raw] for raw in CANONICAL_COLS if raw in df.columns}
    df = df.rename(columns=keep_map)

    # pick canonical columns that actually exist + season
    selected = [c for c in CANONICAL_COLS.values() if c in df.columns] + ["season"]
    df = df[selected]

    # parse date
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True, infer_datetime_format=True)
    df = df.dropna(subset=["date"])
    
    # Harmonize team names
    for col in ["home_team", "away_team"]:
        if col in df.columns:
            df[col] = df[col].map(canonicalize_team)

    # Build a stable match_id if we have the necessary fields
    if all(c in df.columns for c in ["date", "home_team", "away_team"]):
        df = df.sort_values(["date"]).reset_index(drop=True)
        df["match_id"] = (
            df["date"].dt.strftime("%Y%m%d")
            + "_"
            + df["home_team"].str.replace("\s+", "_", regex=True)
            + "_"
            + df["away_team"].str.replace("\s+", "_", regex=True)
        )

    return df


def discover_files(input_root: Path, pattern: str) -> List[Path]:
    files = sorted(input_root.rglob(pattern))
    if not files:
        logger.error(f"No files matched under {input_root} with pattern '{pattern}'")
    else:
        logger.info(f"Discovered {len(files)} file(s)")
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, required=True, help="Root folder containing season subfolders")
    ap.add_argument("--pattern", type=str, default="E0.csv", help="Filename pattern to match (default: E0.csv)")
    ap.add_argument("--out-csv", type=Path, default=Path("data/processed/epl_matches.csv"))
    ap.add_argument("--out-parquet", type=Path, default=Path("data/processed/epl_matches.parquet"))
    ap.add_argument("--season-hint", type=str, default=None, help="Optional explicit season label to apply")
    args = ap.parse_args()

    files = discover_files(args.input_root, args.pattern)
    if not files:
        return

    frames: List[pd.DataFrame] = []
    for f in files:
        try:
            frames.append(load_single_csv(f, args.season_hint))
        except Exception as e:
            logger.exception(f"Failed to load {f}: {e}")

    if not frames:
        logger.error("No data loaded. Exiting.")
        return

    all_df = pd.concat(frames, ignore_index=True, sort=False)

    # Arrange columns (canonical first, then any extras), drop exact duplicates
    preferred_order = [
        "match_id", "season", "date", "home_team", "away_team",
        "full_time_home_goals", "full_time_away_goals", "full_time_result",
        "home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target",
        "home_corners", "away_corners", "home_fouls", "away_fouls",
        "home_yellow", "away_yellow", "home_red", "away_red",
        "b365_home_odds", "b365_draw_odds", "b365_away_odds",
    ]
    cols = [c for c in preferred_order if c in all_df.columns] + [
        c for c in all_df.columns if c not in preferred_order
    ]
    all_df = all_df[cols]

    before = len(all_df)
    all_df = all_df.drop_duplicates()
    logger.info(f"Concatenated {before} rows -> {len(all_df)} unique rows after dropping duplicates")

    # Ensure output folder exists
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)

    # Write outputs
    all_df.to_csv(args.out_csv, index=False)
    try:
        all_df.to_parquet(args.out_parquet, index=False)
    except Exception as e:
        logger.warning(f"Parquet write failed (pyarrow missing?): {e}")

    # Quick summary to console
    logger.info(f"Saved CSV -> {args.out_csv}")
    logger.info(f"Saved Parquet -> {args.out_parquet}")
    if {"season", "date"}.issubset(all_df.columns):
        logger.info(
            "Time span: %s to %s",
            all_df["date"].min().date() if pd.api.types.is_datetime64_any_dtype(all_df["date"]) else "?",
            all_df["date"].max().date() if pd.api.types.is_datetime64_any_dtype(all_df["date"]) else "?",
        )
    if {"home_team", "away_team"}.issubset(all_df.columns):
        n_teams = pd.unique(pd.concat([all_df["home_team"], all_df["away_team"]])).size
        logger.info("Distinct teams: %d", n_teams)


if __name__ == "__main__":
    main()
