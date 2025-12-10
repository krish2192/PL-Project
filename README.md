# EPL Match Outcome Predictor

End-to-end pipeline to predict English Premier League match outcomes (Home / Draw / Away) from historical results, rolling form, Elo, and betting odds. Includes ingestion, feature engineering, model training, a FastAPI service, and a lightweight web client.

## Highlights
- Ingests football-data.co.uk season CSVs, normalizes column names, harmonizes team aliases, and builds stable match IDs.
- Leakage-safe feature engineering (rolling form, rest days, Elo with home advantage, implied probabilities from odds).
- Time-aware cross-validation with a baseline multinomial logistic regression; optional LightGBM for stronger performance.
- FastAPI backend exposes /health, /teams, /predict, and a handy /limits endpoint; static web UI consumes the API.
- Works locally; ready to drop into a container or CI to refresh data weekly.

## Quickstart
```bash
# 0) Python env
python -m venv .venv
. .venv/Scripts/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -U pandas numpy pyarrow scikit-learn joblib fastapi uvicorn[standard] lightgbm

# 1) Ingest raw season files (e.g., data/raw/epl/**/E0.csv from football-data.co.uk)
python data_ingest.py --input-root data/raw/epl --pattern "E0.csv" \
  --out-csv data/processed/epl_matches.csv --out-parquet data/processed/epl_matches.parquet

# 2) Build pre-match features (rolling stats, rest, Elo, implied odds)
python build_features.py --input data/processed/epl_matches.parquet --out data/processed/epl_features.parquet

# 3a) Train baseline logistic regression
python train_baseline.py --features data/processed/epl_features.parquet --model-out models/baseline_logreg.pkl

# 3b) (Optional) Train LightGBM
python train_lgbm.py --features data/processed/epl_features.parquet --model-out models/lgbm.pkl

# 4) Serve API (reload for dev)
uvicorn api_server:app --reload --port 8000

# 5) Try it
curl "http://127.0.0.1:8000/health"
curl "http://127.0.0.1:8000/teams"
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" \
  -d '{"home_team":"Arsenal","away_team":"Chelsea","date":"2025-08-12"}'

# 6) Web client
# Open web/index.html in a browser (or use a simple static server) and set the API Base URL.
```

## Repo Map
- data_ingest.py ? load raw football-data.co.uk CSVs, normalize schema, add season labels, team alias cleanup, stable match_id.
- build_features.py ? leakage-safe rolling form, Elo (home advantage configurable), rest days, implied probabilities from odds; saves Parquet + CSV.
- train_baseline.py ? multinomial logistic regression with time-series CV; saves sklearn pipeline to models/baseline_logreg.pkl.
- train_lgbm.py ? LightGBM multiclass with time-series CV; saves to models/lgbm.pkl.
- api_server.py ? FastAPI service that rebuilds features on-the-fly for a given date and teams, then predicts via the trained pipeline.
- web/index.html + web/main.js ? minimal UI to hit the API and visualize probabilities.
- data/raw/epl/ ? place source CSVs here (not tracked).
- data/processed/ ? ingested and feature-engineered outputs (generated).
- models/ ? saved pipelines (generated).

## Feature Engineering Cheatsheet
- Rolling windows: goals for/against, shots, shots on target, points over configurable windows (default 3/5/10), shifted to avoid leakage.
- Rest & scheduling: days since last match per team; rest_diff between home and away.
- Elo: simple Elo with configurable K and home-advantage offset; ratings are pre-match values.
- Betting odds: Bet365 odds mapped to implied probabilities (if present).
- Target: y = {0: Home, 1: Draw, 2: Away}.

## Modeling Notes
- Uses TimeSeriesSplit to respect chronology; metrics: accuracy, log loss, macro F1 (printed per fold and averaged).
- Pipelines include imputers and one-hot encoders to handle missing odds/early-season gaps and unseen teams/seasons.
- You can calibrate probabilities (e.g., isotonic) and save a bundle as models/lgbm_calibrated.pkl; api_server prefers that if present.

## API Surface
- GET /health ? basic status
- GET /teams ? list known teams
- GET /limits ? min/max allowable dates (based on data)
- POST /predict ? body: {home_team, away_team, date}; returns probabilities for Home/Draw/Away and top_class label

## What to Showcase on a Resume
- Built an end-to-end sports betting/modeling stack: ingestion ? feature engineering (rolling form, Elo, rest) ? time-aware CV ? FastAPI ? web UI.
- Emphasized leakage prevention and chronological validation for realistic performance estimates.
- Served calibrated probability outputs with a lightweight client for stakeholder demos.

## Next Improvements (nice-to-have)
- Add MLflow/W&B tracking for experiments.
- Dockerfile + docker-compose to run API + web.
- Unit tests for feature builders and /predict happy-path.
- Scheduled refresh to pull latest season data weekly and retrain.
